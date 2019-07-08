#Font:  https://matplotlib.org/3.1.0/gallery/pie_and_polar_charts/pie_features.html
#       https://pythonspot.com/matplotlib-pie-chart/

#######################
# ----- Imports ----- #
#######################

# matplotlib
import matplotlib.pyplot as pyplot

def generateForAllMutants():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = '', 'Minimais', 'Equivalentes'
    sizes = [85.358969836, 6.826600811, 7.814429353]
    explode = (0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    # Set colors
    blue700 = '#1976D2'
    orange700 = '#FFA000'
    red700 = '#E64A19'

    colors = [red700, blue700, orange700]

    fig1, ax1 = pyplot.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    pyplot.show()

def generateForMinimals():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Minimais', 'Minimais em arcos \nprimitivos'
    sizes = [41.86046512, 58.13953488]
    explode = (0, 0.035)  # only "explode" the 2nd slice (i.e. 'Hogs')

    # Set colors
    blue700 = '#1976D2'
    blue200 = '#90CAF9'

    colors = [blue200, blue700]

    fig1, ax1 = pyplot.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    pyplot.show()

def generateForEquivalents():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Equivalentes', 'Equivalentes em arcos \nprimitivos'
    sizes = [38.8261851, 61.1738149]
    explode = (0, 0.035)  # only "explode" the 2nd slice (i.e. 'Hogs')

    # Set colors
    orange700 = '#FFA000'    
    orange200 = '#FFE082'

    colors = [orange200, orange700]

    fig1, ax1 = pyplot.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    pyplot.show()

def generateForMinimalsOrEquivalents():
    # Data to plot
    labels = ['Minimais', 'Equivalentes']
    sizes = [387, 443]
    labels_gender = ['Arco Primitivo', ' ', 'Arco Primitivo', ' ']
    sizes_gender = [225, 162, 271, 172]
    
    # Set colors
    blue700 = '#1976D2'
    blue200 = '#90CAF9'
    orange700 = '#FFA000'
    orange200 = '#FFE082'
    white = '#FFFFFF'

    colors = [blue700, orange700]
    colors_gender = [blue200, white, orange200, white]
    
    # Plot
    patches = pyplot.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%', pctdistance=0.875, frame=True)
    pyplot.pie(sizes_gender, colors=colors_gender, radius=0.75, autopct='%1.1f%%', pctdistance=0.85, startangle=90)
    centre_circle = pyplot.Circle((0,0), 0.5, color='black', fc='white', linewidth=0)
    fig = pyplot.gcf()
    fig.gca().add_artist(centre_circle)
    
    pyplot.axis('equal')
    pyplot.tight_layout()
    pyplot.show()     

def generateWithSubPlotExample():
# Data to plot
    labels = ['Python', 'C++', 'Ruby', 'Java']
    sizes = [30, 25, 35, 10]
    labels_gender = ['Man', 'Woman', 'Man', 'Woman', 'Man', 'Woman', 'Man', 'Woman']
    sizes_gender = [20, 10, 10, 15, 30, 5, 7, 3]
    colors = ['#ff6666', '#ffcc99', '#99ff99', '#66b3ff']
    colors_gender = ['#c2c2f0', '#ffb3e6', '#c2c2f0', '#ffb3e6', '#c2c2f0', '#ffb3e6', '#c2c2f0', '#ffb3e6']
    
    # Plot
    pyplot.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%', pctdistance=0.85, frame=True)
    pyplot.pie(sizes_gender, colors=colors_gender, radius=0.75, autopct='%1.1f%%', pctdistance=0.85, startangle=90)
    centre_circle = pyplot.Circle((0,0), 0.5, color='black', fc='white', linewidth=0)
    fig = pyplot.gcf()
    fig.gca().add_artist(centre_circle)
    
    pyplot.axis('equal')
    pyplot.tight_layout()
    pyplot.show() 

if __name__ == '__main__':
    #generateForAllMutants()
    generateForMinimals()
    generateForEquivalents()
    #generateForMinimalsOrEquivalents()
    #generateWithSubPlotExample()