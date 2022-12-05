# with and without measurement
for b in 0 1
do
  # increasing number of qbits
  for i in {0..4}
  do
    python experiment_scalability.py -g 0 -q "$i" -m "$b" >> experiment_depth.log
  done
done

for b in 0 1
do
  # increasing number of gates
  for i in {0..13}
  do
    python experiment_scalability.py -g "$i" -q 0 -m "$b" >> experiment_width.log
  done
done

