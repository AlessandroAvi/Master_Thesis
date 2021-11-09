import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os



ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
confusion_matrix = np.zeros((10,10))
labels = ['0','1','2','3','4','5','6','7','8','9']
size = len(labels)




# -------- READ FROM TXT FILE
with open(ROOT_PATH + '\\training_results.txt') as f:

    j,i = 0,0
    for line in f:

        data = line.split(',')
        for number in data:
            confusion_matrix[j,i] = float(number)
            i+=1

        j+=1
        i=0






# -------- CREATE BAR PLOT
blue2 = 'cornflowerblue'
colors = [blue2]*size
bar_values = np.zeros(size)

for i in range(0, size):
    bar_values[i] = round(round(confusion_matrix[i,i]/sum(confusion_matrix[i,:]),4)*100,2)

fig = plt.subplots(figsize =(12, 8))
bar_plot = plt.bar(labels, bar_values, color=colors, edgecolor='grey')

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
plt.xticks([r for r in range(size)], labels, fontweight ='bold', fontsize = 12)
plt.title('Accuracy test - Method used: '  , fontweight ='bold', fontsize = 15)
plt.show()






# -------- CREATE CONFUSION MATRIX
figure = plt.figure()
axes = figure.add_subplot()

caxes = axes.matshow(confusion_matrix, cmap=plt.cm.Blues)
figure.colorbar(caxes)

for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        axes.text(x=j, y=i,s=int(confusion_matrix[i, j]), va='center', ha='center', size='large')

axes.xaxis.set_ticks_position("bottom")
# The 2 following lines generate and error - I was not able to solve that but is not problematic
axes.set_xticklabels([''] + labels)
axes.set_yticklabels([''] + labels)

plt.xlabel('PREDICTED LABEL', fontsize=10)
plt.ylabel('TRUE LABEL',      fontsize=10)
plt.title('Confusion Matrix', fontsize=15, fontweight ='bold')
plt.show()
