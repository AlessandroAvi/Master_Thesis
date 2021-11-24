import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os



ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
confusion_matrix = np.zeros((10,10))
openmv_labels = []

# -------- READ FROM TXT FILE
with open(ROOT_PATH + '\\training_results.txt') as f:

    j,i = 0,0
    label_flag = 0
    for line in f:  # cycle over lines 

        if(label_flag==0):
            data = line.split(',')  # split one line in each single number
            for number in data:
                openmv_labels.append(number)
            label_flag = 1
            confusion_matrix = np.zeros((len(openmv_labels),len(openmv_labels)))
        else:   

            data = line.split(',')  # split one line in each single number
            for number in data:
                confusion_matrix[j,i] = float(number)   # save the number
                i+=1

            j+=1
            i=0
# --------

size = len(openmv_labels)



# -------- CREATE BAR PLOT
blue2 = 'cornflowerblue'
colors = [blue2]*size
bar_values = np.zeros(size)

for i in range(0, size):
    bar_values[i] = round(round(confusion_matrix[i,i]/sum(confusion_matrix[i,:]),4)*100,2)

fig = plt.subplots(figsize =(12, 8))
bar_plot = plt.bar(openmv_labels, bar_values, color=colors, edgecolor='grey')

# Add text to each bar showing the percent
for p in bar_plot:
    height = p.get_height()
    xy_pos = (p.get_x() + p.get_width() / 2, height)
    xy_txt = (0, -20) 

    # Avoid the text to be outside the image if bar is too low
    if(height>10):
        plt.annotate(str(height), xy=xy_pos, xytext=xy_txt, textcoords="offset points", ha='center', va='bottom', fontsize=12)
    else:
        plt.annotate(str(height), xy=xy_pos, xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)


# Plot
plt.ylim([0, 100])
plt.ylabel('Accuracy %', fontsize = 15)
plt.xlabel('Classes', fontsize = 15)
plt.xticks([r for r in range(size)], openmv_labels, fontweight ='bold', fontsize = 12)
plt.title('Accuracy test - Method used: '  , fontweight ='bold', fontsize = 15)
plt.show()
# --------





# -------- CREATE CONFUSION MATRIX
fig, ax = plt.subplots()
im = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
fig.colorbar(im)

# Loop over data dimensions and create text annotations.
for i in range(len(openmv_labels)):
    for j in range(len(openmv_labels)):
        text = ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", size='large')

ax.set_xticks(np.arange(len(openmv_labels)))
ax.set_yticks(np.arange(len(openmv_labels)))
ax.xaxis.set_ticks_position("bottom")
# NB THAT THE DIGITS LABELS ARE NEVER PUT, JUST USE THE DEFAULT LABEL NOTATION FROM 0 TO 9

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
fig.tight_layout()
plt.show()
# --------