This program has a python file, an excel spread sheet, and these instructions.
Please leave these files in the same folder.

PREREQUISITES
You will need to install Python 3.
Once you have python, you need to pip install numpy, pandas, and scipy
You also need Microsoft Excel


IMPORTING YOUR SPLITS
In livesplit, go to Share > Export To Excel, then save your splits as an Excel File, and open them
Then, go to the tab called "Segment History - Real Time"
In a separate tab, open the Excel file "input.xlsx".

In the region that is colored yellow, paste in your split names
In the region colored green, write in the probability that you reset in this level because of a run-ending mistake (e.g "10" for 10%
	(you can esitmate this yourself, or you can look at your run on splits.io to see your true reset rate, and copy those in)
In the region colored orange, enter your goal time
In the region colored white, copy in your segment splits. Each column corresponds to one split.
	(you need to manually delete segments that don't represent that split correctly, e.g. if you skipped the previous split)


RUNNING THE PROGRAM
Open file explorer and navigate to the folder containing this program. Type "cmd" into the 
address bar (without the quotes to open up the terminal)
In the command window, type "python prob.py input.xlsx > output.txt" without the quotes and hit enter.
After a few minutes, the program will finish. The output will be in a file called "output.txt"


READING THE REPORT
The report has multiple parts. The optimal reset policy is at the bottom. It also has all of your
splits sorted by reset rate and standard devation. These are likely the more important splits in the run.
Finally, it also tells you the probability of getting your goal time, and how many hours you would have
to play on average to get your goal time.


WARNING
The output is only as good as the data you give it.