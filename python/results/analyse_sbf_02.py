import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import scipy
from sklearn.metrics import mean_squared_error

from lib.util.env import getbase_dir
import matplotlib.ticker as ticker


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
    fig.savefig(getbase_dir(['results','sbf_02b']) + "zz_all_ds_considering_split_number.png", dpi=300)

def plot_all_ds_considering_percent(df,dash_styltes):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    # sns.lineplot(data=df, x='x', y='mean_dist_of_real',hue='ds',style='ds',dashes=dash_styles)
    sns.lineplot(data=df, x='x', y='mean_dist_of_real', hue='ds' , dashes=[(2, 2)])

    # q1 = lambda p: np.exp(6.999 - .7903 * np.log(p))
    #
    # sns.set_style("whitegrid")
    # q1 = lambda p:  -.7903 * np.log(p)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # q2 = lambda p: -1*np.log(0.34*p)
    # q3 = lambda p: 1/(1 + np.log(p)) #-1.69314718/1
    # q3 = lambda p: 1 / (1 + p * np.log(p))  # -1.69314718/1
    # q3 = lambda p: 1 / (1 + p)  # -1.69314718/1
    # P = np.linspace(0.0, 0.5, num=10)
    # # print(P)
    # q3(P)
    # ax.plot(P, q2(P), color="BLUE", lw=3, label='Q2')
    # ax.plot(P, q1(P), color="RED", lw=3, label='Q1')
    # ax.plot(P, q3(P), color="GREEN", lw=3, label='Q1')
    # plt.show()

    #ax.set_xscale('log')
    # ax.xaxis.set_major_locator(ticker.LogLocator(base=2.0, subs=(1.0, ), numdecs=0, numticks=None))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    plt.title("SBF Split Error")
    plt.ylabel("Mean Error (\u03B5)")
    plt.xlabel("Split length in % of original filter")

    plt.show()
    fig.savefig(getbase_dir(['results','sbf_02b']) + "zz_all_ds_error_bit_percent.png",dpi=300)

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

def plot_error_epsilon_distribution(z,marcas=[1, 3, 5, 6],fs=(12, 6)):
    # z = df[df.bf_type == 'BBF']
    # z = df
    z['nd'] = (z.full - z.psim_mean)*100
    # zz = z[z.x <= 0.2]
    # zz = zz[zz.x >= 0.1]

    #fig, ax = plt.subplots()

    #ax.violinplot(zz.nd, vert=True)
    #sns.distplot(zz.nd, fit=st.laplace, kde=False)
    # Show the plot
    #plt.show()

    # labels = list(z.x.unique()[[1, 2, 3, 4, 5, 6]])
    labels = list(z.x.unique()[marcas])
    fig, axes = plt.subplots(2, int(len(labels) / 2), figsize=fs,
                             constrained_layout=True)
                             # sharex=True)
    # fig.subplots_adjust(top=0.8)
    fig.suptitle("\u03B5-Error Distribution",y=1.05)

    colors = ['skyblue', 'olive', 'gold', 'purple', 'teal', 'red']
    # labels = z.x.unique()

    for x in range(0, len(labels)):
        print(x)
        eixo = False
        if x < len(labels) / 2:
            eixo = axes[0, x]
            # axes[x]
        else:
            eixo = axes[1, int(x - len(labels) / 2)]

        eixo.set_title('Split of {:.2%}'.format(labels[x]))
        # sns.distplot(z[z.x == labels[x]].nd, fit=st.laplace , color=colors[x],
        sns.distplot(z[z.x == labels[x]].nd, fit=st.laplace,
                     label='length={}'.format(x), kde=False,
                     ax=eixo)
        # ax=axes[0,x])

    for ax1 in axes.flat:
        # ax1.set(xlabel='x-label', ylabel='y-label')
        ax1.set(xlabel='error')
    plt.show()
    fig.savefig(getbase_dir(['results', 'sbf_02b']) + "zz_p_error_episilon.png", dpi=300)



def exponential_regression2var_v1(func_exp,x_data, y_data,eq_label=r'$f(x) = {:.2f} * ln( {:.2f} * x)$'):
    """
    Original

    :param func_exp:
    :param x_data:
    :param y_data:
    :param eq_label:
    :return:
    """
    fig = plt.gcf()
    popt, pcov = scipy.optimize.curve_fit(func_exp, x_data, y_data, p0 = (-1, 0.01))
    print(popt)
    puntos = plt.plot(x_data, y_data, 'x', color='xkcd:maroon', label = "data")

    y_predicted = func_exp(x_data, *popt)
    rmse = np.sqrt(mean_squared_error(y_data, y_predicted))

    eq_label = eq_label.format(*popt) + ", rmse = {:.3f}".format(rmse)
    curva_regresion = plt.plot(x_data, func_exp(x_data, *popt), color='xkcd:teal', label = eq_label)

    # curva_regresion = plt.plot(x_data, func_exp(x_data, *popt), color='xkcd:teal', label=eq_label + end_label)
    plt.legend()
    plt.title("Estimated Error in SBF")
    plt.xlabel("$x=\\frac{s}{l}$")
    plt.ylabel('Error')
    plt.show()
    fig.savefig(getbase_dir(['results', 'sbf_02b']) + "new_estimated_sbf_erro.png", dpi=300)
    # plt.close()

    return popt
    # return func_exp(x_data, *popt),popt

def exponential_regression2var(func_exp,x_data, y_data, xg, yg , eq_label=r'$f(x) = {:.2f} * ln( {:.2f} * x) + {:.2f}$'):
    # func_exp = q2
    # x_data = X
    # y_data = y
    # xg = Xg
    fig = plt.gcf()
    popt, pcov = scipy.optimize.curve_fit(func_exp, x_data, y_data, p0 = (-1, 0.01, 0))
    print(popt)
    puntos = plt.plot(xg, yg, 'x', color='xkcd:maroon', label = "data")

    y_predicted = func_exp(xg, *popt)
    rmse = np.sqrt(mean_squared_error(yg, y_predicted))

    eq_label = eq_label.format(*popt) + ", rmse = {:.3f}".format(rmse)
    curva_regresion = plt.plot(x_data, func_exp(x_data, *popt), color='xkcd:teal', label = eq_label)

    # curva_regresion = plt.plot(x_data, func_exp(x_data, *popt), color='xkcd:teal', label=eq_label + end_label)
    plt.legend()
    plt.title("Estimated Error in SBF")
    plt.xlabel("$x=\\frac{s}{l}$")
    plt.ylabel('Error')
    plt.show()
    fig.savefig(getbase_dir(['results', 'sbf_02b']) + "new_estimated_sbf_erro.png", dpi=300)
    # plt.close()

    return popt

def exponential_regression(q3,q2,x_data, y_data):
    popt3, pcov3 = scipy.optimize.curve_fit(q3, x_data, y_data, p0=(-10, 0.01, 1))
    popt2, pcov2 = scipy.optimize.curve_fit(q2, x_data, y_data, p0=(-1, 0.01))
    # popt1, pcov1 = scipy.optimize.curve_fit(q1, x_data, y_data, p0=(-1, 0.01))

    puntos = plt.plot(x_data, y_data, 'x', color='xkcd:maroon', label="data")

    c3 = plt.plot(x_data, q3(x_data, *popt3), color='xkcd:blue',
                  label="f(x) = a * ln(b * x) + c \n a={:.2f}, b={:.2f}, c={:.2f}".format(*popt3))
    c2 = plt.plot(x_data, q2(x_data, *popt2), color='xkcd:green',
                  label="f(x) = a * ln(b * x) \n a={:.2f}, b={:.2f}".format(*popt2))
    # c1 = plt.plot(x_data, q1(x_data, *popt1), color='xkcd:red',
    #               label="f(x) = ln(b * x),   a={:.2f}, b={:.2f}".format(*popt2))

    plt.legend()
    plt.show()

def plot_episilon_approximation(a,b):
    a=-0.042876301194393125
    b=3.2574724870013103
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(5, 4))
    fe = lambda x, a , b: -1 * np.log(a * np.log(b * x)) #eq completa
    fe1 = lambda x, a, b: np.log( 1/ (a * np.log(b * x)) )
    simplificada_01 = lambda x, a, b: np.log(a) - np.log(b * x)
    as1 = lambda x: ( np.log(1/np.log(1/x)) ) + 3.7 # aqui
    # as2 = lambda x: np.log(1/np.log(1 / x))
    # as2 = lambda x: np.log(1 / np.log(x))
    # as2 = lambda x,a: np.log(-1*np.abs(a)*np.log(1/x))
    # as2(p)
    # as2 = lambda x,a: 1 / np.log(a*np.log(1 / x))
    # as3 = lambda x, b: 1 / np.log( (np.log(1/b)+np.log(1 / x)) )
    # assintotico = lambda x: -1 / np.log(x)

    p = np.linspace(0.00001, .25, num=50)

    data = []
    for xp in p:
        data.append((xp,fe(xp, a, b),r'$ln(\frac{1}{a * ln(b*x)})$'))
        # data.append((xp, fe(xp, -1, 1), r'$ln1(\frac{1}{a * ln(b*x)})$'))
                     # "$\\frac{1}{a * ln(b*x)}$"))
        # data.append((xp,fe1(xp, a, b), 'c1'))
        # data.append((xp, simplificada_01(xp, 1, b), 's1'))
        data.append((xp, as1(xp), r'$ln(\frac{1}{ln(\frac{1}{x})})+ c ,  c=2$'))
        # data.append((xp, as2(xp,-1*a), 'as2'))
        # data.append((xp, as2(xp,a), 'as2'))
        # data.append((xp, as3(xp, b), 'as3'))
    labels = ['x', 'y', 'function']

    r = pd.DataFrame.from_records(data, columns=labels)
    sns.lineplot(data=r,x='x',y='y',hue='function')

    plt.title("$\\epsilon\ estimation$")
    plt.xlabel("splits size$(\\frac{s}{l})$")
    plt.ylabel(r'$\epsilon$')

    plt.show()
    plt.close()
    fig.savefig(getbase_dir(['results', 'sbf_02b']) + "episilon_estimation.png", dpi=400)




if __name__ == 'main':
    import sys

    # from decimal import *
    # getcontext().prec = 2
    df = readResultsSPF02()

    # normality test

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


    # bbf_df = df[df.bf_type == 'BBF']


    z = dataExtract_exp01(df)
    # z[z.bf_type == 'BBF'].mean_erro = 0.02
    # z[z.bf_type == 'XBF'].mean_erro = z[z.bf_type == 'XBF'].mean_erro + 0.004
    # z[z.bf_type == 'BLIP'].mean_erro = z[z.bf_type == 'BLIP'].mean_erro + 0.006
    # z[z.splits < 128]
    # z  = z
    plot_all_ds_considering_split_number(z[z.splits < 128].round(2),dash_styles)

    df['x'] = ((df.orignal_bits_size / df.splits) / df.orignal_bits_size)
    plot_all_ds_considering_percent(df[df.bf_type == 'BBF'], dash_styles)

    ### person testando a correlação entre o teste
    import scipy
    z = df.copy()
    z['nd'] = abs(z.full - z.psim_mean)

    pr , pv = scipy.stats.pearsonr(z.nd,z.mean_dist_of_real)
    scipy.stats.pearsonr(z.full, z.full-z.mean_dist_of_real)
    print(pr,pv)

    plot_error_epsilon_distribution(df,marcas=[1, 3, 4, 5],fs=(8,6))

    # REGRESSAO PARA MOSTRAR A EQUACAO QUE CASA COM O ERRO
    df['x'] = ((df.orignal_bits_size / df.splits) / df.orignal_bits_size)
    df.x.unique()
    bdf = df[(df.bf_type == 'BBF')]
    bdf = df

    X = []
    y = []
    for x in bdf.x.unique():
        if x != 0.5:
            X.append(x)
            y.append(bdf[bdf.x == x].mean_dist_of_real.mean())

    X = np.asarray(X)
    y = np.asarray(y)
    # X = np.asarray(bdf[['x']]).T[0]
    #y = np.asarray(bdf[['mean_dist_of_real']]).T[0]

    ## plotar com dados individualizados
    Xg = []
    yg = []
    for ids in bdf.ds.unique():
        tz = bdf[bdf.ds == ids]
        for x in bdf[bdf.ds == ids].x.unique():
            if x != 0.5 and x > 0.01:
                Xg.append(x)
                yg.append(tz[tz.x == x].mean_dist_of_real.mean())

    Xg = np.asarray(Xg)
    yg = np.asarray(yg)

    # best -0.042, 0.088 , -0.142
    # best -0.042, 0.088 , -0.142
    # c depois
    q2 = lambda x, a, b , c : a * np.log(b * x) + c
    a,b,c = exponential_regression2var(q2, X, y , Xg , yg)

    q2 = lambda x, a, b: a * np.log(b * x)
    a, b = exponential_regression2var_v1(q2, Xg, yg)

    plot_episilon_approximation(a, b)

    sys.exit()

    for ds in df.ds.unique():
        print(ds + "; {} ; {}".format(len(df[df.ds == ds].id_a.unique()), len(df[df.ds == ds].id_b.unique())))

    from lib.mybloom.bloomfilter import BloomFilter

    b = BloomFilter(cap=96)

    #gabarito
    bases = ['bikes','beer', 'books1', 'eletronics', 'movies1', 'music', 'restaurants1']




    df = df[df.id_a != 'ltable._id']
    df = df[df.id_b != 'rtable._id']
    df = df.round(2)

    for datadir in df.ds.unique():
        dsg = df[(df.ds == datadir) & (df.bf_type == 'BBF')]
        base_dir = getbase_dir(['Datasets', datadir ])  # + os.sep
        gab_files = base_dir + 'labeled_data.csv'
        print(gab_files)

dsg.id_a = pd.to_numeric(dsg.id_a)
dsg.id_b = pd.to_numeric(dsg.id_b)

gs = pd.read_csv(gab_files,skiprows=5)

r0 = []
r1 = []
for index, row in dsg.iterrows():
    aid = row.id_a
    bid = row.id_b
    if len(gs[(gs['ltable._id'] == aid) & (gs['rtable._id'] == bid) & (gs.gold == 1)]) == 1:
        s1 = row.sbf_sim
        s2 = row.full
        r0.append(row.splits,s2)
    else:
        r1.append(row.splits,row.full)


out = {'splits': i, 'similarity': 'all-splits', 'bf_type' : j
                    ,'mean_erro': np.mean(temp.exp_01),
                   'std': np.std(temp.exp_01)
                   }
            z.append(out)


np.median(r0),np.mean(r0),np.std(r0),np.max(r0),np.min(r0),len(r0)
np.median(r1),np.mean(r1),np.std(r1),np.max(r1),np.min(r1),len(r1)
np.mean(r1)

writer = pd.ExcelWriter('pandas_multiple.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df1.to_excel(writer, sheet_name='Sheet1')
df2.to_excel(writer, sheet_name='Sheet2')
df3.to_excel(writer, sheet_name='Sheet3')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

for index, row in gs[gs.gold == 1].iterrows():
    aid = row['ltable._id']
    bid = row['rtable._id']
    dsg[(dsg.id_a == aid) & (dsg.id_b == bid)]
    df[(df.id_a == aid) & (df.id_b == bid)]




    # sns.lineplot(x=p,y=fe(p,a,b))

    # sns.lineplot(x=p, y=simplificada_01(p, 1, b))
    # plt.show()
    # plt.close()

    ###
    # fig = plt.gcf()
    # popt, pcov = scipy.optimize.curve_fit(func_exp, x_data, y_data, p0=(-1, 0.01))
    # print(popt)



    # q3 = lambda x, a, b, c: a * np.log(b * x) + c
    # q2 = lambda x, a, b: a + np.log(b * x)
    #
    # exponential_regression(q3,q2, X, y)


























    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # sns.lineplot(data=df, x='x', y='mean_dist_of_real',hue='ds',style='ds',dashes=dash_styles)
    sns.lineplot(data=df, x='x', y='mean_dist_of_real', hue='ds', dashes=[(2, 2)])





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

