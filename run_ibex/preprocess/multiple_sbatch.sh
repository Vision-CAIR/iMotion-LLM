#!/bin/bash

# for i in {0..99}
# do
#    sbatch /home/felembaa/projects/iMotion-LLM-ICLR/run_ibex/preprocess/train_gf_17feb_2025.sh $i
# done

# for i in {0..14}
# do
#    sbatch /home/felembaa/projects/iMotion-LLM-ICLR/run_ibex/preprocess/valid_gf_17feb_2025.sh $i
# done

# for i in {1..99}
# do
#    sbatch /home/felembaa/projects/iMotion-LLM-ICLR/run_ibex/preprocess/train_gf_to_mtr_mapping_28nov.sh $i
# done

# for i in {1..99}
# do
#    sbatch /home/felembaa/projects/iMotion-LLM-ICLR/run_ibex/preprocess/train_gf_14sep_2025.sh $i
# done



# for i in {0..99}
# do
#    sbatch /home/felembaa/projects/iMotion-LLM-ICLR/run_ibex/preprocess/train_gf_to_mtr_mapping_28nov.sh $i
# done

# for i in {0..99}
# do
#    sbatch /home/felembaa/projects/iMotion-LLM-ICLR/run_ibex/preprocess/train_gf_30sep.sh $i
# done

# for i in {0..14}
# do
#    sbatch /home/felembaa/projects/iMotion-LLM-ICLR/run_ibex/preprocess/valid_gf_30sep.sh $i
# done

# for i in {0..14}
# do
#    sbatch /home/felembaa/projects/iMotion-LLM-ICLR/run_ibex/preprocess/nuplan_preprocess_complex.sh $i
# done

# for i in {1..13}
# do
#    sbatch /home/felembaa/projects/iMotion-LLM-ICLR/run_ibex/preprocess/call_chatgpt.sh $i
# done



# for i in {0..14}
# do
#    sbatch /home/felembaa/projects/iMotion-LLM-ICLR/run_ibex/preprocess/valid_gf_30sep.sh $i
# done

for i in {0..14}
do
   sbatch /home/felembaa/projects/iMotion-LLM-ICLR/run_ibex/preprocess/valid_2agent_complementary.sh $i
done
