#!/bin/sh

############################ single case ############################
#case_dir='./examples/case2/'
#case_dir='/Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained2.jl/test/data/Network_02O-173/scenario_9/'
##case_dir='/Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained2.jl/test/data/c1/scenario_1/'
#raw=$case_dir'case.raw'
#rop=$case_dir'case.rop'
#con=$case_dir'case.con'
#inl=$case_dir'case.inl'
#sol1=$case_dir'sol1/sol1_1_3.txt'
#sol2=$case_dir'sol2/sol2_1_3.txt'
#summary=$case_dir'summary.csv'
#detail=$case_dir'detail_1_3.csv'
#
## run it
#python test.py "$raw" "$rop" "$con" "$inl" "$sol1" "$sol2" "$summary" "$detail"



############################## evaluate all scenarios for a given network ################################
#network_dir='/Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained2.jl/test/data/Network_02O-173/'
#network_dir='/Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained2.jl/test/data/c2/'
#for dir in $network_dir/*/;
#
#do
#    case_dir=$network_dir$(basename -- $dir)/
#    echo case_dir
#    raw=$case_dir'case.raw'
#    rop=$case_dir'case.rop'
#    con=$case_dir'case.con'
#    inl=$case_dir'case.inl'
#    sol1=$case_dir'solution1_111.txt'
#    sol2=$case_dir'solution2_v2.txt'
#    summary=$case_dir'summary_v2.csv'
#    detail=$case_dir'detail_v2.csv'
#
#    python test.py "$raw" "$rop" "$con" "$inl" "$sol1" "$sol2" "$summary" "$detail"
#done




############################## create sol1, sol2 folders and move all sol1_* files to sol1 folder ###########
#network_dir='/Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained2.jl/test/data/Network_02O-173/'
#cd $network_dir
#for dir in */;
#do
#  cd $network_dir$dir
#  # if folder sol1 does not exists
#  if [ ! -d "sol1" ]; then
#    mkdir sol1
#  fi
#  if [ ! -d "sol2" ]; then
#    mkdir sol2
#  fi
#
#  if [ ! -d "detail" ]; then
#    mkdir detail
#  fi
#  if [ ! -d "summary" ]; then
#    mkdir summary
#  fi
#
#  if ls sol1_* 1> /dev/null 2>&1; then
#    mv sol1_* sol1/
#  fi
#done


############################# evaluate all sol2 and sol1 and generate detail.csv files in the detail folder ###########
network_dir='/Users/xiongjinxin/A-xjx/SRIBD/PowerModelsSecurityConstrained2.jl/test/data/Network_02O-173/'
# get the current working directory into evaluation_dir
evaluation_dir=$(pwd)

cd $network_dir
for scenario in */;

do
  ## for test: if scenario is not "scenario_9" continue
  if [ $scenario != "scenario_15/" ]; then
    continue
  fi

#  echo $scenario
  cd $network_dir$scenario
  if [ ! -d "detail" ]; then
    mkdir detail
  fi
  if [ ! -d "summary" ]; then
    mkdir summary
  fi
  raw=$network_dir$scenario'case.raw'
  rop=$network_dir$scenario'case.rop'
  con=$network_dir$scenario'case.con'
  inl=$network_dir$scenario'case.inl'
  # for all files in sol1 folder
  for file1 in sol1/*;
  # get the last part of the file name
  do
#    echo $file1
    # get the last part of the file name
    file1_name=$(basename -- $file1)
#    echo $file1_name
    # get the first part of the file name
    file1_name=${file1_name%.*}
#    echo $file1_name
    # get the part of file name without sol1_
    code=${file1_name#*_}
#    echo $code
    # get the file name of sol2
    file2="sol2/sol2_$code.txt"
#    echo $file2
    # get the file name of detail
    detail="detail/detail_$code.csv"
#    echo $detail
    # get the file name of summary
    summary="summary/summary_$code.csv"
#    echo $summary
    # run the test.py

    python $evaluation_dir"/test.py" "$raw" "$rop" "$con" "$inl" "$file1" "$file2" "$summary" "$detail"
  done
done