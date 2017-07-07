# Get script path
SCRIPT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"

# Set path for current session
export PATH=$PATH:$SCRIPT_PATH
echo "$SCRIPT_PATH appended to PATH."

# Set path permanently
CONFIG_FILE=~/.bash_profile
echo >> $CONFIG_FILE
echo "# added by mripy installer" >> $CONFIG_FILE
echo "export \"PATH=\$PATH:$SCRIPT_PATH\"" >> $CONFIG_FILE

# Setup ipython for mac's default python 2.7
source setup_ipython.sh
