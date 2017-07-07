# Usage: source ~/PythonPlus/sources/mripy/scripts/switch_python.csh default
# This is the only way to affect the current shell session

# Find anaconda path
# set PYTHON3_PATH = "/Users/qcc/PythonPlus/anaconda/bin"
set CONFIG_DIR = ~/.mripy
if ( ! -d $CONFIG_DIR ) then
    mkdir $CONFIG_DIR
endif
set CONFIG_FILE = $CONFIG_DIR/python3_path
if ( ! -e $CONFIG_FILE ) then
    set PYTHON3_PATH = `find ~ -iregex ".*/anaconda/bin"`
    echo $PYTHON3_PATH > $CONFIG_FILE
else
    set PYTHON3_PATH = `cat $CONFIG_FILE`
endif
set VER = $1

# Change $PATH
if ( $VER == "default" ) then
    setenv PATH ` echo $PATH | sed -e "s|${PYTHON3_PATH}:||g" `
    echo "Switched to default Python (presumably Python 2.7 at /usr/bin)"
else
    setenv PATH ${PYTHON3_PATH}:$PATH
    echo "Switched to anaconda (presumably Python 3 at $PYTHON3_PATH)"
endif

# echo $PATH
