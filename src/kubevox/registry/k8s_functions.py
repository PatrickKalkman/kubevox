"""
Definitions of functions that interact with a Kubernetes cluster that the LLM uses to execute commands.
"""

import os
import re
from collections import defaultdict
from typing import Any, Dict, Optional

import aiohttp
import yaml
from kubernetes import client, config

from kubevox.registry.function_registry import FunctionRegistry


@FunctionRegistry.register(
    description="Get the number of nodes in the Kubernetes cluster.",
    response_template="The cluster has {node_count} nodes.",
)
async def get_number_of_nodes() -> Dict[str, Any]:
    """Get the total number of nodes in the cluster."""
    config.load_kube_config()
    v1 = client.CoreV1Api()
    nodes = v1.list_node()
    return {"node_count": len(nodes.items)}


@FunctionRegistry.register(
    description="Get the number of pods in the Kubernetes cluster, optionally filtered by namespace.",
    response_template="There are {pod_count} pods running{namespace_info}.",
    parameters={
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": (
                    "Namespace to filter pods (optional - if not provided, counts pods across all namespaces)"
                ),
            },
        },
    },
)
async def get_number_of_pods(namespace: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the total number of pods, optionally filtered by namespace.

    Args:
        namespace: Optional namespace to filter pods. If None, counts pods across all namespaces.
    """
    config.load_kube_config()
    v1 = client.CoreV1Api()

    if namespace:
        pods = v1.list_namespaced_pod(namespace=namespace)
        namespace_info = f" in namespace '{namespace}'"
    else:
        pods = v1.list_pod_for_all_namespaces()
        namespace_info = " across all namespaces"

    return {"pod_count": len(pods.items), "namespace_info": namespace_info}


@FunctionRegistry.register(
    description="Get the number of namespaces in the Kubernetes cluster.",
    response_template="The cluster contains {namespace_count} namespaces.",
)
async def get_number_of_namespaces() -> Dict[str, Any]:
    """Get the total number of namespaces in the cluster."""
    config.load_kube_config()
    v1 = client.CoreV1Api()
    namespaces = v1.list_namespace()
    return {"namespace_count": len(namespaces.items)}


@FunctionRegistry.register(
    description="Analyze logs from all pods in a deployment for criticals/errors/warnings in the last hour.",
    response_template="Analysis complete for deployment '{deployment_name}' in namespace '{namespace}'.",
    parameters={
        "type": "object",
        "properties": {
            "deployment_name": {
                "type": "string",
                "description": "Name of the deployment to analyze.",
            },
            "namespace": {
                "type": "string",
                "description": "Namespace of the deployment (default: 'default').",
                "default": "default",
            },
        },
        "required": ["deployment_name"],
    },
)
async def analyze_deployment_logs(deployment_name: str, namespace: str = "default") -> Dict[str, Any]:
    """
    Analyze logs from all pods in a deployment.

    Args:
        deployment_name: Name of the deployment
        namespace: Namespace of the deployment (default: "default")

    Returns:
        Dict containing analysis results
    """
    config.load_kube_config()
    v1 = client.CoreV1Api()

    # Get pods for deployment
    pods = v1.list_namespaced_pod(namespace=namespace, label_selector=f"app={deployment_name}")

    log_analysis = defaultdict(int)
    for pod in pods.items:
        try:
            logs = v1.read_namespaced_pod_log(name=pod.metadata.name, namespace=namespace, since_seconds=3600)

            # Count occurrences
            log_analysis["CRITICAL"] += logs.count("CRITICAL")
            log_analysis["ERROR"] += logs.count("ERROR")
            log_analysis["WARNING"] += logs.count("WARNING")

        except Exception as e:
            print(f"Error getting logs for pod {pod.metadata.name}: {str(e)}")

    return {
        "deployment_name": deployment_name,
        "namespace": namespace,
        "log_counts": dict(log_analysis),
    }


@FunctionRegistry.register(
    description="Get version information for both Kubernetes API server and nodes.",
    response_template="API server version is {api_version}. Node versions: {node_versions}.",
)
async def get_version_info() -> Dict[str, Any]:
    """Get version information for the Kubernetes cluster."""
    config.load_kube_config()
    v1 = client.CoreV1Api()
    version = client.VersionApi().get_code()

    nodes = v1.list_node()
    node_versions = [node.status.node_info.kubelet_version for node in nodes.items]

    return {"api_version": version.git_version, "node_versions": node_versions}


@FunctionRegistry.register(
    description="Retrieve the latest stable version information from the Kubernetes GitHub repository.",
    response_template="Latest Kubernetes stable version is {latest_stable_version}.",
)
async def get_kubernetes_latest_version_information() -> Dict[str, Any]:
    """Get the latest stable Kubernetes version from GitHub."""
    url = "https://raw.githubusercontent.com/kubernetes/kubernetes/master/CHANGELOG/CHANGELOG-1.28.md"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()

    # Extract version using regex
    version_match = re.search(r"# v1\.28\.(\d+)", content)
    if version_match:
        latest_version = f"1.28.{version_match.group(1)}"
    else:
        latest_version = "Unknown"

    return {"latest_stable_version": latest_version}


@FunctionRegistry.register(
    description="Get a list of all available Kubernetes clusters from the kubeconfig.",
    response_template="Found {total_clusters} clusters. Active cluster is '{active_cluster[name]}'.",
)
async def get_available_clusters() -> Dict[str, Any]:
    """Get information about available Kubernetes clusters."""
    kubeconfig = os.path.expanduser("~/.kube/config")
    with open(kubeconfig, "r") as f:
        config_data = yaml.safe_load(f)

    clusters = []
    current_context = config_data.get("current-context")

    for cluster in config_data.get("clusters", []):
        cluster_info = {
            "name": cluster["name"],
            "server": cluster["cluster"]["server"],
            "is_active": cluster["name"] == current_context,
        }
        clusters.append(cluster_info)

    active_cluster = next((c for c in clusters if c["is_active"]), None)

    return {
        "clusters": clusters,
        "total_clusters": len(clusters),
        "active_cluster": active_cluster,
    }


@FunctionRegistry.register(
    description="Switch to a different Kubernetes cluster context and persist the change.",
    response_template="Switched to cluster '{cluster_name}'.",
    parameters={
        "type": "object",
        "properties": {
            "cluster_name": {
                "type": "string",
                "description": "Name of the cluster to switch to.",
            },
        },
        "required": ["cluster_name"],
    },
)
async def switch_cluster(cluster_name: str) -> Dict[str, Any]:
    """
    Switch to a different cluster context.

    Args:
        cluster_name: Name of the cluster to switch to

    Returns:
        Dict containing the result of the operation
    """
    config.load_kube_config()

    try:
        # Use kubectl command through os.system
        result = os.system(f"kubectl config use-context {cluster_name}")
        success = result == 0

        return {
            "cluster_name": cluster_name,
            "success": success,
            "error": None if success else "Failed to switch context",
        }
    except Exception as e:
        return {"cluster_name": cluster_name, "success": False, "error": str(e)}


@FunctionRegistry.register(
    description="Get the name of the current Kubernetes cluster.",
    response_template="Current cluster is '{cluster_name}'.",
)
async def get_cluster_name() -> Dict[str, str]:
    """Get the name of the current cluster context."""
    config.load_kube_config()
    contexts, active_context = config.list_kube_config_contexts()
    return {"cluster_name": active_context["name"]}


@FunctionRegistry.register(
    description="Retrieve the messages of the last four events in the cluster.",
    response_template="Retrieved the last {count} events from the cluster.",
)
async def get_last_events(count: int = 4) -> Dict[str, Any]:
    """
    Get the last N events from the cluster.

    Args:
        count: Number of events to retrieve (default: 4)

    Returns:
        Dict containing the events
    """
    config.load_kube_config()
    v1 = client.CoreV1Api()

    events = v1.list_event_for_all_namespaces(limit=count)

    event_list = []
    for event in events.items:
        event_list.append(
            {
                "type": event.type,
                "reason": event.reason,
                "message": event.message,
                "timestamp": event.last_timestamp,
            }
        )

    return {"events": event_list, "count": len(event_list)}


@FunctionRegistry.register(
    description="Get detailed status information about the Kubernetes cluster.",
    response_template="Cluster status retrieved. Summary: {status_summary}.",
)
async def get_cluster_status() -> Dict[str, Any]:
    """Get comprehensive status information about the cluster."""
    config.load_kube_config()
    v1 = client.CoreV1Api()

    # Get nodes status
    nodes = v1.list_node()
    node_status = defaultdict(int)
    for node in nodes.items:
        for condition in node.status.conditions:
            if condition.type == "Ready":
                node_status[condition.status] += 1

    # Get pods status
    pods = v1.list_pod_for_all_namespaces()
    pod_status = defaultdict(int)
    for pod in pods.items:
        pod_status[pod.status.phase] += 1

    status_summary = (
        f"{len(nodes.items)} nodes "
        f"({node_status['True']} ready), "
        f"{len(pods.items)} pods "
        f"({pod_status['Running']} running)"
    )

    return {
        "node_status": dict(node_status),
        "pod_status": dict(pod_status),
        "status_summary": status_summary,
    }
