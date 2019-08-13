import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lib.util.env import getbase_dir


def plotResult(ds,save=True):
    df = pd.read_csv(getbase_dir('results')+ds+'.csv',sep=';')
    df = df[df.bf_type != 'BBLIP']


    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data=df,x='p',y='diff',hue='bf_type', showfliers=False)
    plt.title("")
    plt.ylabel("Diferença em %")
    # plt.show()
    fig.savefig(getbase_dir('results')+'sns_bp_'+ds+'.png')



ax = sns.lmplot(data=df,x='p',y='diff',hue='bf_type',
                fit_reg=False, height=10, x_estimator=np.mean, truncate=True)
ax.plot(range(len(mean_val)), mean_val, '-b')
plt.show()

##############

def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                      transform=ax.transAxes, color='gray')
    line.set_clip_on(False)
    ax.add_line(line)

def label_len(my_index,level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k,g in groupby(labels)]

def label_group_bar_table(ax, df):
    ypos = -.1
    scale = 1./df.index.size
    for level in range(df.index.nlevels)[::-1]:
        pos = 0
        for label, rpos in label_len(df.index,level):
            lxpos = (pos + .5 * rpos)*scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
            add_line(ax, pos*scale, ypos)
            pos += rpos
        add_line(ax, pos*scale , ypos)
        ypos -= .1


###################

from matplotlib import pyplot as plt

def mk_groups(data):
    try:
        newdata = data.items()
    except:
        return

    thisgroup = []
    groups = []
    for key, value in newdata:
        newgroups = mk_groups(value)
        if newgroups is None:
            thisgroup.append((key, value))
        else:
            thisgroup.append((key, len(newgroups[-1])))
            if groups:
                groups = [g + n for n, g in zip(newgroups, groups)]
            else:
                groups = newgroups
    return [thisgroup] + groups

def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                      transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)

def label_group_bar(ax, data):
    groups = mk_groups(data)
    xy = groups.pop()
    x, y = zip(*xy)
    ly = len(y)
    xticks = range(1, ly + 1)

    ax.boxplot(xticks, y, align='center')
    ax.set_xticks(xticks)
    ax.set_xticklabels(x)
    ax.set_xlim(.5, ly + .5)
    ax.yaxis.grid(True)

    scale = 1. / ly
    for pos in xrange(ly + 1):
        add_line(ax, pos * scale, -.1)
    ypos = -.2
    while groups:
        group = groups.pop()
        pos = 0
        for label, rpos in group:
            lxpos = (pos + .5 * rpos) * scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
            add_line(ax, pos * scale, ypos)
            pos += rpos
        add_line(ax, pos * scale, ypos)
        ypos -= .1

if __name__ == '__main__':
    data = {'Room A':
               {'Shelf 1':
                   {'Milk': 10,
                    'Water': 20},
                'Shelf 2':
                   {'Sugar': 5,
                    'Honey': 6}
               },
            'Room B':
               {'Shelf 1':
                   {'Wheat': 4,
                    'Corn': 7},
                'Shelf 2':
                   {'Chicken': 2,
                    'Cow': 1}
               }
           }
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    label_group_bar(ax, data)
    fig.subplots_adjust(bottom=0.3)
    fig.savefig('label_group_bar_example.png')