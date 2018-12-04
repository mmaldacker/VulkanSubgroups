import csv
import pygal
from collections import defaultdict

reduceChart = pygal.Line(logarithmic=True, title='Reduce', y_title='Time (us)', x_title='Number of elements')
scanChart = pygal.Line(logarithmic=True, title='Scan', y_title='Time (us)', x_title='Number of elements')

reduceDict = defaultdict(list)
scanDict = defaultdict(list)

sizes = []

with open('timing.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        name = row['name'].split('/')
        if name[0] == 'Reduce_GPU_Subgroup':
            sizes.append(name[1])
        if name[0].startswith('Reduce'):
            reduceDict[name[0].replace('Reduce_', '').replace('_', ' ')].append(row['real_time'])
        else:
            scanDict[name[0].replace('Scan_', '').replace('_', ' ')].append(row['real_time'])

for name, values in reduceDict.items():
    reduceChart.add(name, [float(value) / 1000.0 for value in values])

reduceChart.x_labels = sizes
reduceChart.render_to_file('reduce.svg')

for name, values in scanDict.items():
    scanChart.add(name, [float(value) / 1000.0 for value in values])

scanChart.x_labels = sizes
scanChart.render_to_file('scan.svg')