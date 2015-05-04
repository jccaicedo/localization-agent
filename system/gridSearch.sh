
# aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor
for expName in trial2 trial3; do
  sleep 4000
  for cost in 0.0001; do
    for overlap in 0.3; do
      for category in aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor ;  do
        qsub -N job-$category-$cost-$overlap-$expName experiment.sh ;
      done
    done
  done
  echo "Waiting"
done
