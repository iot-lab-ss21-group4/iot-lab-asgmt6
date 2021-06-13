param(
    [string] $DeploymentType = "edge",
    [string] $AppName = "forecaster",
    [string] $EnvFilePath = ".env.secret.edge.forecaster",
    [Parameter(Mandatory = $true)] [string] $DockerId
)

# Load the build arguments for 'docker build' and the corresponding Dockerfile
$build_args = Get-Content -Raw -Path $EnvFilePath | ConvertFrom-StringData
# Convert all keys to lowercase letters before adding as a build argument
$build_arg_str = ""
foreach ($key in $build_args.PSBase.Keys) {
    $build_arg_str += " --build-arg $($key.ToString().ToLowerInvariant())=$($build_args[$key])"
}

$dockerfile_path = "docker-apps/${DeploymentType}/${AppName}/Dockerfile"
$build_context_path = "."
$build_tag = "${DockerId}/iotlab:${DeploymentType}-${AppName}"
$docker_build_cmd = "docker build -f ${dockerfile_path} ${build_context_path} -t ${build_tag}"
$docker_build_cmd += $build_arg_str
Invoke-Expression -Command $docker_build_cmd
docker push ${build_tag}
