# Get package path
PACKAGE_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" ; cd ..; pwd -P )"

rm -r "/Volumes/FTP/sample programs/qcc/mripy"
cp -r $PACKAGE_PATH "/Volumes/FTP/sample programs/qcc/mripy"
