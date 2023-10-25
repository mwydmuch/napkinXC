DATASET=wiki10
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 100"
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 3 --covWeights 1"
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 5 --covWeights 1"
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 5 --labelsWeights invP"
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 5 --labelsWeights invPs"
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 5 --labelsWeights wLog"
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 5 --labelsWeights wPow"

bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 100 --threads 1"
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 3 --covWeights 1 --threads 1"
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 5 --covWeights 1 --threads 1"
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 5 --labelsWeights invP --threads 1"
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 5 --labelsWeights invPs --threads 1"
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 5 --labelsWeights wLog --threads 1"
bash ./experiments/test.sh ${DATASET} "-m plt --seed 13" "--topK 5 --labelsWeights wPow --threads 1"
bash ./experiments/test_bc.sh ${DATASET} "-m plt --seed 13" "--topK 3"
bash ./experiments/test_bc.sh ${DATASET} "-m plt --seed 13" "--topK 5"

