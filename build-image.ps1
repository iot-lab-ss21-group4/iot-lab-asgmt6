param(
    [string] $DeploymentType = "edge",
    [string] $AppName = "forecaster",
    [Parameter(Mandatory = $true)] [string] $DockerId
)

docker build -f docker-apps/${DeploymentType}/${AppName}/Dockerfile . -t ${DockerId}/iotlab:${DeploymentType}-${AppName}
docker push ${DockerId}/iotlab:${DeploymentType}-${AppName}
