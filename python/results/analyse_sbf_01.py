import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

from lib.util.env import getbase_dir
import matplotlib.ticker as ticker


def readResultsSPF01():
    df = pd.read_csv(getbase_dir('results') + 'bikes' + '.csv', sep=';')
    df['ds'] = 'bike'
    df = df[df.bf_type != 'BBLIP']

    for ds in ['beer', 'books1', 'eletronics', 'movies1', 'music', 'restaurants1']:
        bdf = pd.read_csv(getbase_dir('results') + ds + '.csv', sep=';')
        bdf['ds'] = ds
        bdf = bdf[bdf.bf_type != 'BBLIP']
        df = pd.concat([df, bdf])

    return df


####
####
####

def sintetizeResultByDataset(df):
    r = []
    for ds in df.ds.unique():
        for p in df.p.unique():
            for bft in df.bf_type.unique():
                mv = df[(df.ds == ds) & (df.p == p) & (df.bf_type == bft)]['diff'].median()
                me = df[(df.ds == ds) & (df.p == p) & (df.bf_type == bft)]['diff'].mean()
                r.append({'dataset': ds, 'p': p, 'bf_type': bft, 'median_error': mv, 'mean_error': me})

    rdf = pd.DataFrame(r)
    rdf['np'] = rdf['p'].map({'10%': 0.1, '20%': 0.2, '30%': 0.3, '40%': 0.4, '50%': 0.5,
                              '60%': 0.6, '70%': 0.7, '80%': 0.8, '90%': 0.9})

    return rdf


def plot_summaryByDataset(rdf):

    #jitter x
    def f(g):
        return np.random.normal(g, 0.03)

    def g(x):
        return abs(np.random.normal(g, 0.03))

    rdf.p = rdf.np.apply(f)
    # rdf['median_error'] = rdf['median_error'].apply(g)
    # cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.scatterplot(x="p", y="median_error", hue="bf_type",
                         alpha=0.8, x_jitter=True, s=150,

                         style='dataset', palette="Set2",
                         data=rdf)

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.9, -0.1),
                    ncol=3)
    # ax.legend(frameon=True, loc='lower center', ncol=4)
    # text = ax.text(-0.2,1.05, "Aribitrary text", transform=ax.transAxes)
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_title("SBF Error")
    ax.set_ylabel("Median error in %")
    ax.grid('on')
    plt.tight_layout()
    plt.show()
    fig.savefig(getbase_dir('results') + 'erro_in_all_ds_.png')


def plot_sbfErrorInDataset(df,domain):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='p', y='diff', hue='bf_type', showfliers=False)
    plt.title("SBF Error in " + domain + "Dataset")
    plt.ylabel("Error in %")
    plt.show()
    fig.savefig(getbase_dir('results') + "sbf_error_ds_" + domain + ".png")


def plot_sbfError(df):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='p', y='diff', hue='bf_type', showfliers=False)
    plt.title("SBF Error")
    plt.ylabel("Error in %")
    plt.show()
    fig.savefig(getbase_dir('results') + "sbf_error_all_ds" + ".png")

####
#### Parts
####


def arrangePartsData(pdf):
    df = pdf.copy()
    df['temp'] = abs(df.full - df.p1)
    p1 = df[['p', 'bf_type' , 'ds' ,'temp']]
    p1['part'] = 'left'
    del df['temp']

    df['temp'] = abs(df.full - df.p2)
    p2 = df[['p', 'bf_type' , 'ds' ,'temp']]
    p2['part'] = 'rigth'
    del df['temp']
    del df['id_a']
    del df['id_b']
    return pd.concat([p1, p2])

def plot_erroInSBFParts(pdf):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=pdf, x='ds', y='temp', hue='part',
                showfliers=False, notch=True)

    plt.title("SBF Error in splits")
    plt.ylabel("Error in %")
    plt.show()
    fig.savefig(getbase_dir('results') + "sbf_parts_error_.png")



####
#### Evaluation of parts
####

def readResultsSPF02():
    df = pd.read_csv(getbase_dir(['results','sbf_02_data']) + 'msplit_' +'bikes' + '.csv', sep=';')
    df['ds'] = 'bike'
    df = df[df.bf_type != 'BBLIP']

    for ds in ['beer', 'books1', 'eletronics', 'movies1', 'music', 'restaurants1']:
        bdf = pd.read_csv(getbase_dir(['results','sbf_02_data']) + 'msplit_' + ds + '.csv', sep=';')
        bdf['ds'] = ds
        bdf = bdf[bdf.bf_type != 'BBLIP']
        df = pd.concat([df, bdf])

    return df


def dataExtract_exp01(bbf_df):
    """
    Processa os resultados e agrupa por similarade (todos os splits eq-01 ou um split eq-2)
    :param bbf_df: o dataset com os resultados
    :return:
    """
    bbf_df = bbf_df.round(2)
    bbf_df['exp_01'] = bbf_df.full - bbf_df.sbf_sim

    z = []
    for i in bbf_df.splits.unique():
        for j in bbf_df.bf_type.unique():
            # temp = bbf_df[(bbf_df.splits == i) & (bbf_df.ds == j)]
            temp = bbf_df[(bbf_df.splits == i) & (bbf_df.bf_type == j)]

            # if i < 129:
            # temp = bbf_df[(bbf_df.splits == i)]
            # bbf_df['exp_01_std'] = np.std(temp.exp_01)

            out = {'splits': i, 'similarity': 'all-splits', 'bf_type' : j
                    ,'mean_erro': np.mean(temp.exp_01),
                   'std': np.std(temp.exp_01)
                   }
            z.append(out)

            # eq1 mean
            out = {'splits': i, 'similarity': 'one-split','bf_type' :  j,
                   'mean_erro': np.mean(temp.mean_dist_of_real),
                   'std': np.std(temp.mean_dist_of_real)
                   }
            z.append(out)

            # out = {'splits': i, 'ds': 'eq2-max',
            #        'mean_erro': np.mean(temp.max_dist_of_real),
            #        'std': np.std(temp.max_dist_of_real)
            #        }
            # z.append(out)

            # out = {'splits': i, 'ds': 'eq2-min',
            #        'mean_erro': np.mean(temp.min_dist_of_real),
            #        'std': np.std(temp.min_dist_of_real)
            #        }
            # z.append(out)

    return pd.DataFrame(z).round(2)


def plot_all_ds_considering_split_number(df,dash_styltes):
    """
    Equation 01 of section 2

    :param df:
    :param dash_styltes:
    :return:
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=df, x='splits', y='mean_erro',hue='bf_type',style='similarity',dashes=dash_styles)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.xaxis.set_major_locator(ticker.LogLocator(base=2.0, subs=(1.0, ), numdecs=0, numticks=None))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # ax.axhline(0.01, ls='--')

    plt.title("Similarity Error")
    # plt.ylabel("Error (\u03B5)")
    plt.ylabel("Error")
    plt.xlabel("Number of splits")

    plt.show()
    fig.savefig(getbase_dir(['results','sbf_02b']) + "zz_all_ds_considering_split_number.png")


def plot_all_ds_considering_percent(df,dash_styltes):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    # sns.lineplot(data=df, x='x', y='mean_dist_of_real',hue='ds',style='ds',dashes=dash_styles)
    sns.lineplot(data=df, x='x', y='mean_dist_of_real', hue='ds' , dashes=[(2, 2)])

    q1 = lambda p: np.exp(6.999 - .7903 * np.log(p))

    sns.set_style("whitegrid")
    q1 = lambda p:  -.7903 * np.log(p)
    fig, ax = plt.subplots(figsize=(10, 6))
    q2 = lambda p: -1*np.log(0.34*p)
    q3 = lambda p: 1/(1 + np.log(p)) #-1.69314718/1
    q3 = lambda p: 1 / (1 + p * np.log(p))  # -1.69314718/1
    q3 = lambda p: 1 / (1 + p)  # -1.69314718/1
    P = np.linspace(0.0, 0.5, num=10)
    # print(P)
    q3(P)
    ax.plot(P, q2(P), color="BLUE", lw=3, label='Q2')
    ax.plot(P, q1(P), color="RED", lw=3, label='Q1')
    ax.plot(P, q3(P), color="GREEN", lw=3, label='Q1')
    plt.show()

    #ax.set_xscale('log')
    # ax.xaxis.set_major_locator(ticker.LogLocator(base=2.0, subs=(1.0, ), numdecs=0, numticks=None))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    plt.title("SBF Split Error")
    plt.ylabel("Mean Error (\u03B5)")
    plt.xlabel("Split length in % of original filter")

    plt.show()
    fig.savefig(getbase_dir('results') + "zz_all_ds_error_bit_percent.png")

# def plot_all_ds_considering_bits(df,dash_styltes):
#     sns.set_style("whitegrid")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.lineplot(data=df, x='bits', y='median_dist_of_real',hue='ds',style='ds',dashes=dash_styles)
#     ax.set_xscale('log')
#     ax.xaxis.set_major_locator(ticker.LogLocator(base=2.0, subs=(1.0, ), numdecs=0, numticks=None))
#     ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
#
#     plt.title("SBF Split Error")
#     plt.ylabel("Median Error")
#     plt.xlabel("Split length in bits")
#
#     plt.show()
#     fig.savefig(getbase_dir('results') + "zz_all_ds_error_bit_size.png")


if __name__ == 'main':
    import sys

    sys.exit()

    from decimal import *
    getcontext().prec = 3


    #sbf_02b
    dash_styles = ["",
                   (4, 1.5),
                   (1, 1),
                   (3, 1, 1.5, 1),
                   (5, 1, 1, 1),
                   (5, 1, 2, 1, 2, 1),
                   (2, 2, 3, 1.5),
                   (1, 2.5, 3, 1.2),
                   (2, 1),
                   (4, 1.5),
                   (3, 1, 2.5, 1),
                   (5, 1.3, 1.5, 1),
                   ]

    df = readResultsSPF02()
    # bbf_df = df[df.bf_type == 'BBF']


    z = dataExtract_exp01(df)
    # z  = z
    plot_all_ds_considering_split_number(z[z.splits < 512].round(2),dash_styles)

    df['x'] = ((df.orignal_bits_size / df.splits) / df.orignal_bits_size)
    plot_all_ds_considering_percent(df[df.bf_type == 'BBF'], dash_styles)


    ####
    #### Regressions
    ####


    X = np.asarray(df[['x']])
    y = np.asarray(df[['mean_dist_of_real']])

    ##
    ## non linear
    ##
    from sklearn.svm import SVR

    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.01)

    svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
                   coef0=1)

    lw = 2

    svrs = [svr_rbf, svr_lin, svr_poly]
    kernel_label = ['RBF', 'Linear', 'Polynomial']
    model_color = ['m', 'c', 'g']

    ig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    for ix, svr in enumerate(svrs):
        axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
                      label='{} model'.format(kernel_label[ix]))
        axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                         edgecolor=model_color[ix], s=50,
                         label='{} support vectors'.format(kernel_label[ix]))
        axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         facecolor="none", edgecolor="k", s=50,
                         label='other training data')
        axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)

    fig.text(0.5, 0.04, 'data', ha='center', va='center')
    fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()


    ##
    ## linear
    ##
    np.polyfit(np.asarray(df[['x']]).T[0],np.asarray(df[['mean_dist_of_real']]).T[0], 1)

    from sklearn import linear_model

    reg = linear_model.Lasso(alpha=0.1)
    lasso = reg.fit(np.asarray(df[['x']]),np.asarray(df[['mean_dist_of_real']]))

    reg = linear_model.Ridge(alpha=0)
    ridge = reg.fit(np.asarray(df[['x']]), np.asarray(df[['mean_dist_of_real']]))

    reg = linear_model.LogisticRegression()
    lr = reg.fit(np.asarray(df[['x']]), np.asarray(df[['mean_dist_of_real']]))


    print_coefficients(ridge)

    X = np.asarray(df[['x']])
    y_pred = ridge.predict(X)
    # res = np.log(Y - y_pred)
    plt.plot(X, y_pred, 'k.', color='blue', )
    plt.show()

    def print_coefficients(model):
        w = list(model.coef_)
        w.reverse()
        print(np.poly1d(w) + model.intercept_)



    #
    # plot_all_ds_considering_bits(df, dash_styles)
    #
    #

    sys.exit()































    # main

    df = readResultsSPF01()
    sdf = sintetizeResultByDataset(df)
    plot_summaryByDataset(sdf)

    for ds in df.ds.unique():
        plot_sbfErrorInDataset(df[df.ds == ds], ds)

    plot_sbfError(df)

    # parts

    pdf = arrangePartsData(df)
    plot_erroInSBFParts(pdf)

df = readResultsSPF02()

# df = pd.read_csv(getbase_dir('results') + 'msplit_bikes' + '.csv', sep=';')
df['y'] = abs(df.full - df.sbf_sim)*100
df['bits'] = df.orignal_bits_size / df.splits
df['x'] = ((df.orignal_bits_size / df.splits) / df.orignal_bits_size)



filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
filled_markers = (".","o","P","h","x",",","D",",","s")

dash_styles = ["",
               (4, 1.5),
               (1, 1),
               (3, 1, 1.5, 1),
               (5, 1, 1, 1),
               (5, 1, 2, 1, 2, 1),
               (2, 2, 3, 1.5),
               (1, 2.5, 3, 1.2),
               (2, 1),
               (4, 1.5),
               (3, 1, 2.5, 1),
               (5, 1.3, 1.5, 1),
               ]


plot_all_ds_considering_bits(df,dash_styles)
plot_all_ds_considering_percent(df,dash_styles)




######################################################################################################

df['py'] = abs(df.full - df.psim_median)
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='splits', y='py')
#ax.set_xscale('log')
#ax.set_yscale('log')
#plt.axvline(7, 0.05 ,3,color='red')
#ax.set(xscale="log")
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
plt.title("SBF Split Error in ")
plt.ylabel("Error")
plt.show()
fig.savefig(getbase_dir('results') + "zz_erro_incease_split.png")



sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df.head(50000), x='bits', y='median_dist_of_real',hue='ds',style='ds',dashes=dash_styles)
ax.set_xscale('log')
# ax.set_xticklabels(rotation=30)
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
#ax.get_xaxis().get_major_formatter().set_scientific(False)
#ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.xticks(2**np.arange(10, dtype = np.uint64))
ax.set_xlim(4,4096)
plt.title("SBF Split Error")
plt.ylabel("Error")
plt.locator_params(axis='x', nbins=9)
plt.setp(ax.get_xticklabels(), rotation=30)
plt.show()
#fig.savefig(getbase_dir('results') + "zz_bits_def.png")


####

