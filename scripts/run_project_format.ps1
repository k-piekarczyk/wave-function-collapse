Write-Output "Running [ isort ]"
isort .
if (!$?) {
    Write-Output ""
    exit
}

Write-Output ""
Write-Output "Running [ black ]"
black .
if (!$?) {
    Write-Output ""
    exit
}

Write-Output ""
Write-Output "Running [ mypy ]"
mypy .
if (!$?) {
    Write-Output ""
    exit
}

Write-Output ""
Write-Output "Running [ flake8 ]"
flake8 .
if (!$?) {
    Write-Output ""
    exit
}
