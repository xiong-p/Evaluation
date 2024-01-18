#!/bin/sh
cd "/home/jinxin/xjx/SRIBD/PowerModelsSecurityConstrained.jl/src/scripts/c1" || exit

# network_dir='/home/jinxin/xjx/SRIBD/Network_1/'
network_dir='/home/jxxiong/A-xjx/IEEE14/'
for scenario_id in $(seq 3 3);
do
    echo $scenario_id
    scenario=$network_dir"scenario_""$scenario_id""/"
    InFile1=$scenario'case.con'
    InFile2=$scenario'case.inl'
    InFile3=$scenario'case.raw'
    InFile4=$scenario'case.rop'
    NetworkModel="scenario_"$scenario_id
    echo $scenario

    julia --project='~/xjx/SRIBD/PowerModelsSecurityConstrained.jl' -e "include(\"MyJulia1.jl\"); MyJulia1(\"${InFile1}\", \"${InFile2}\", \"${InFile3}\", \"${InFile4}\", 600, 2, \"${NetworkModel}\")"

    julia --project='~/xjx/SRIBD/PowerModelsSecurityConstrained.jl' -e "include(\"MyJulia2.jl\"); MyJulia2(\"${InFile1}\", \"${InFile2}\", \"${InFile3}\", \"${InFile4}\", 600, 2, \"${NetworkModel}\")"
done