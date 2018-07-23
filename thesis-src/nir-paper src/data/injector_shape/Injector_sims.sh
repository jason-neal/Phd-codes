# Simulate an injected observation
for TEMP in 2500 3000 3500 3600 3700 3800 3900 4000 4500;
do
	iam_fake_full_stack.py -n 300 -j 4 INJECTORSIMS$TEMP 1 5200 4.5 0.0 $TEMP 5.0 0.0 0 100 --suffix "_comp_$TEMP"
done

