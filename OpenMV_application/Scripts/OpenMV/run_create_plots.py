import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os



ROOT__PATH = os.path.dirname(os.path.abspath(__file__))
SAVE_PLOTS__PATH = ROOT__PATH + '\\Plots_results\\'


confusion_matrix = np.zeros((10,10))
openmv_labels = []
openmv_times = []
real_labels = ['0','1','2','3','4','5','6','7','8','9']

# -------- READ FROM TXT FILE
with open(ROOT__PATH + '\\training_results.txt') as f:

    j,i = 0,0
    label_flag = 0
    times_flag = 0
    for line in f:  # cycle over lines 

        if(label_flag==0):
            data = line.split(',')  # split one line in each single number
            for number in data:
                openmv_labels.append(number)
            label_flag = 1

        elif(times_flag==0 and label_flag==1):
            data = line.split(',')  # split one line in each single number
            for number in data:
                openmv_times.append(number)
            times_flag = 1

        else:   
            data = line.split(',')  # split one line in each single number
            for number in data:
                confusion_matrix[j,i] = float(number)   # save the number
                i+=1

            j+=1
            i=0
# --------

size = len(real_labels)



# -------- CREATE BAR PLOT
blue2 = 'cornflowerblue'
colors = [blue2]*size
bar_values = np.zeros(size)

for i in range(0, size):
    bar_values[i] = round(round(confusion_matrix[i,i]/sum(confusion_matrix[i,:]),4)*100,2)

fig = plt.subplots(figsize =(12, 8))
bar_plot = plt.bar(real_labels, bar_values, color=colors, edgecolor='grey')

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
plt.xticks([r for r in range(size)], real_labels, fontweight ='bold', fontsize = 12)
plt.title('Accuracy test - Method used: '  , fontweight ='bold', fontsize = 15)
plt.savefig(SAVE_PLOTS__PATH + 'barPlot.png')
plt.show()
# --------





# -------- CREATE CONFUSION MATRIX
fig = plt.figure(figsize =(6,6))
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(confusion_matrix, cmap=plt.cm.Blues, interpolation='nearest')
width, height = confusion_matrix.shape


# Loop over data dimensions and create text annotations.
for x in range(width):
    for y in range(height):
        ax.annotate(str(int(confusion_matrix[x,y])), xy=(y, x), ha="center", va="center", size='large')

cb = fig.colorbar(res)
plt.xticks(range(width), real_labels[:width])
plt.yticks(range(height), real_labels[:height])
plt.savefig(SAVE_PLOTS__PATH + 'confusionMatrix.png')
plt.show()

# --------





# -------- PRINT ON SCREEN AVERAGE TIMES
print('\n\n******************************************************')
print('Here are the average times for inference and training')
print(f'    Average frozen model inference time:        {openmv_times[0]}')
print(f'    Average OL model inference + training time: {openmv_times[1]}')
print(f'    Average total time:                         {openmv_times[1]}')
print('******************************************************')