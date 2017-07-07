#!/usr/bin/env bash

for pp in "" /usr/bin; do
    export PATH=$pp:$PATH
    echo -e "\n# Switch environment to"
    python -V

    echo -e "\n# Test case 01"
    cd /Users/qcc/Desktop/LaminarRivalry/comp_loc_epi/raw/S02
    extract_physio.py -l -p 20170108_S18_ODC_S02_CQ_phy -d func*

    echo -e "\n# Test case 02"
    cd /Volumes/FTP/Rawdata/cwliu/2017_attentionlayer/EPI_0.75iso
    extract_physio.py -l -p 20170105_S18_attention_sjb02_yzqian -d 20170105_S18_ATTENTION_SBJ02_YZQIAN -f 6 7 8 9 14

    echo -e "\n# Test case 03"
    cd /Volumes/raid1/7T_yqian/sbj02
    extract_physio.py -l -p physio -d func*
done
