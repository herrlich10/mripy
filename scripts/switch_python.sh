# Usage: source switch_python.sh default
# This is the only way to affect the current shell session

# Find anaconda path
# PYTHON3_PATH="/Users/qcc/PythonPlus/anaconda/bin" # >>> Change me! <<<
CONFIG_DIR=~/.mripy
if [ ! -d $CONFIG_DIR ]; then
    mkdir $CONFIG_DIR
fi
CONFIG_FILE=$CONFIG_DIR/python3_path
if [ ! -f $CONFIG_FILE ]; then
    PYTHON3_PATH=`find ~ -iregex ".*/anaconda/bin"`
    echo $PYTHON3_PATH > $CONFIG_FILE
else
    PYTHON3_PATH=`cat $CONFIG_FILE`
fi
VER=$1 # From command line arg1

# Change $PATH
if [ $VER = "default" ]; then
    export PATH="$( echo $PATH | sed -e "s|$PYTHON3_PATH:||g" )" # g for global substitution, use -e "" to use variables
    echo "Switched to default Python (presumably Python 2.7 at /usr/bin)"
else
    export PATH=$PYTHON3_PATH:$PATH
    echo "Switched to anaconda (presumably Python 3 at $PYTHON3_PATH)"
fi

# echo $PATH
