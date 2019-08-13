    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    from lib.util.env import getbase_dir
    from encrypt import encrypt_data_in_memory
    from splitting_bf.evaluation import split_bf #change method
    from lib.mybloom.bloomutil import jaccard_coefficient
    
    import matplotlib.ticker as ticker
    
    
    
    #
    def isTrueMatch(gs,ida,idb,base_a='idDBLP',base_b='idACM'):
        z = gs[(gs[base_a] == ida) & (gs[base_b] == idb)]

        if len(z) == 1:
            return True
    
        return False
    
    #isTrueMatch(gold_standard_dblp_acm,'journals/sigmod/Dogac02',507340)
    
    def split_ds(df,n):
        for row in df:
            row.append(split_bf(row[1],n))
            row.append(n)
        return pd.DataFrame(df, columns = ['id' , 'bf', 'sbf' , 'n_sbf'])
    
    
    def calculate_sbf_similarity(dfa,dfb,split_pos=0):
        result = []
        for indexa, rowa in dfa.iterrows():
            for indexb, rowb in dfb.iterrows():
                full_bf_sim = jaccard_coefficient(rowa[1],rowb[1])
                # print(rowa[2])
                split_sim = jaccard_coefficient(rowa[2][split_pos],rowb[2][split_pos])
    
                soma = 0
                for i in range(0,len(rowa[2])):
                    soma += jaccard_coefficient(rowa[2][i], rowb[2][i])
    
                result.append({'id_a': rowa[0], 'id_b': rowb[0], 'full_bf': full_bf_sim, 'split_sim': split_sim, 'sbf_sim': soma/len(rowa[2]), 'splits': len(rowa[2])})
    
        return pd.DataFrame(result)
    
    def simulated_sbf_protocol(df,goldstandard,threshold_a,threshold_b=0.02):
        t = threshold_a-threshold_b
        result = []
        count = 0
    
        for index, row in df.iterrows():
            rp = {'id_a': row['id_a'], 'id_b': row['id_b'] , 'sbf_sim': row['sbf_sim'], 'bf_sim': row['full_bf']}
    
            if row['split_sim'] >= t:
                rp['split_sim'] = True
    
                if row['sbf_sim'] >= threshold_a:
                    if isTrueMatch(goldstandard,row['id_a'],row['id_b']):
                        rp['sbf_stat'] = 'TM'
                    else:
                        rp['sbf_stat'] = 'FM'
            else:
                rp['split_sim'] = False
    
    
            if row['full_bf'] >= threshold_a:
                if isTrueMatch(goldstandard, row['id_a'], row['id_b']):
                    rp['bf_stat'] = 'TM' #criar metodo
                else:
                    rp['bf_stat'] = 'FM'  # criar metodo
    
            result.append(rp)
    
            count += 1
            if count % 500000 == 0:
                print(count / len(df) * 100)
    
        return pd.DataFrame(result)
    
    
    # def simulated_sbf_protocol(df,goldstandard,threshold_a,threshold_b=0.02):
    #     t = threshold_a - threshold_b
    #     result = []
    #     ids_a = df.id_a.unique()
    #     ids_b = df.id_b.unique()
    #     count = 0
    #     for ida in ids_a:
    #         for idb in ids_b:
    #             row = df.iloc[df[(df['id_a'] == ida) & (df['id_b'] == idb)]['sbf_sim'].idxmax()]
    #
    #             if len(row) > 0:
    #                 rp = {'id_a': row['id_a'], 'id_b': row['id_b'], 'sbf_sim': row['sbf_sim'], 'bf_sim': row['full_bf']}
    #
    #                 if row['split_sim'] >= t:
    #                     rp['split_sim'] = True
    #
    #                     if row['sbf_sim'] >= threshold_a:
    #                         if isTrueMatch(goldstandard, row['id_a'], row['id_b']):
    #                             rp['sbf_stat'] = 'TM'
    #                         else:
    #                             rp['sbf_stat'] = 'FM'
    #                 else:
    #                     rp['split_sim'] = False
    #
    #                 if row['full_bf'] >= threshold_a:
    #                     if isTrueMatch(goldstandard, row['id_a'], row['id_b']):
    #                         rp['bf_stat'] = 'TM'  # criar metodo
    #                     else:
    #                         rp['bf_stat'] = 'FM'  # criar metodo
    #
    #                 result.append(rp)
    #
    #             count +=1
    #             if count % 100000 == 0:
    #                 print(count/len(df)*100)
    #
    #     return pd.DataFrame(result)
    
    def private_calculate_precision_recall(zdf,technique):
        clean_r = []
        for index, row in zdf.iterrows():
            # if count % 100 == 0:
            #     print(count)
            # count += 1
    
            # id_max = zdf[(zdf['id_a'] == row['id_a']) & (zdf['id_b'] == row['id_b'])]['sbf_sim'].idxmax()
            id_max_a = zdf[(zdf['id_a'] == row['id_a'])][technique].idxmax()
            id_max_b = zdf[(zdf['id_b'] == row['id_b'])][technique].idxmax()
    
            if (id_max_a == id_max_b) and (id_max_a == index):
                # print(row)
                rp = {'id_a': row['id_a'], 'id_b': row['id_b'], 'sbf_sim': row['sbf_sim'], 'bf_sim': row['bf_sim'],
                      'sbf_stat': row['sbf_stat'],'bf_stat': row['bf_stat']}
                clean_r.append(rp)
    
        return pd.DataFrame(clean_r)
    
    def calculate_precision_recall(x,gs,ds_name,alfa):
        fdf = x[(x.sbf_stat == 'TM') | (x.sbf_stat == 'FM') | (x.bf_stat == 'TM') | (x.bf_stat == 'FM')]
        # fdf.to_csv('fdf.csv',index=False)
        zdf = fdf[fdf.split_sim == True]
        # print(len(zdf))
        count = 0
        clean_sbf = private_calculate_precision_recall(zdf,'sbf_sim')
        clean_bf = private_calculate_precision_recall(fdf,'bf_sim')
    
        a = str(alfa)
        filename = "intermediate_"+ds_name+"_"+a+".csv"
        print(filename)
        fdf.to_csv(getbase_dir(['results', 'sbf_03', filename]))
    
        p_sbf = len(clean_sbf[clean_sbf.sbf_stat == 'TM']) / (
                    len(clean_sbf[clean_sbf.sbf_stat == 'FM']) + len(clean_sbf[clean_sbf.sbf_stat == 'TM']))
        r_sbf = len(clean_sbf[clean_sbf.sbf_stat == 'TM']) / len(gs)
    
        p_bf = len(clean_bf[clean_bf.bf_stat == 'TM']) / (
                len(clean_bf[clean_bf.bf_stat == 'FM']) + len(clean_bf[clean_bf.bf_stat == 'TM']))
        r_bf = len(clean_bf[clean_bf.bf_stat == 'TM']) / len(gs)
    
        return (p_sbf,r_sbf),(p_bf,r_bf)


    def calculate_precision_recall_precomputed(fdf, gs, ds_name, alfa):
        # fdf.to_csv('fdf.csv',index=False)
        zdf = fdf[fdf.split_sim == True]
        # print(len(zdf))
        count = 0

        # clean_sbf = private_calculate_precision_recall(zdf, 'sbf_sim')
        clean_sbf = zdf
        print("\t [ x ] sbf done!")
        # clean_bf = private_calculate_precision_recall(fdf, 'bf_sim')
        clean_bf = fdf
        print("\t [ x ] bf done!")

        p_sbf = len(clean_sbf[clean_sbf.sbf_stat == 'TM']) / \
                (
                len(clean_sbf[clean_sbf.sbf_stat == 'FM']) + len(clean_sbf[clean_sbf.sbf_stat == 'TM'])
                )
        r_sbf = len(clean_sbf[clean_sbf.sbf_stat == 'TM']) / len(gs)

        p_bf = len(clean_bf[clean_bf.bf_stat == 'TM']) / (
                len(clean_bf[clean_bf.bf_stat == 'FM']) + len(clean_bf[clean_bf.bf_stat == 'TM']))
        r_bf = len(clean_bf[clean_bf.bf_stat == 'TM']) / len(gs)

        return (p_sbf, r_sbf), (p_bf, r_bf)


    def plot_line_metric(dfm, title, ds='dblp_acm', dir='sbf_03'):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=pf, x='threshold_alfa', y='vals', hue='bf_type',
                     linewidth=2.5, dashes=[(0, 0), (2, 2)])
        # sns.barplot(data=pf, x='threshold_alfa', y='vals', hue='cols',
        #             linewidth=2.5)
        # palette = "tab10"
        ax.set_title(title)
        ax.set_ylabel("")
        ax.grid('on')
        plt.tight_layout()
        plt.show()
        file = getbase_dir(['results', dir, ds + title + '.png'])
        fig.savefig(file)


if __name__ == 'main':

    gold_standard_dblp_acm = pd.read_csv(getbase_dir(['Datasets', 'leipzig_dblp_acm', 'DBLP-ACM_perfectMapping.csv']),
                                         sep=',')

    acm = pd.read_csv(getbase_dir(['Datasets', 'leipzig_dblp_acm', 'ACM.csv']), sep=',')
    dblp = pd.read_csv(getbase_dir(['Datasets', 'leipzig_dblp_acm', 'DBLP2.csv']), sep=',', encoding='latin')

    eacm = encrypt_data_in_memory(acm, [1, 2, 3, 4], 1024)
    edblp = encrypt_data_in_memory(dblp, [1, 2, 3, 4], 1024)

    df_acm = split_ds(eacm, 4)
    df_dblp = split_ds(edblp, 4)

    z = calculate_sbf_similarity(df_dblp,df_acm)

    similarity = pd.read_csv(getbase_dir(['results', 'sbf_03', 'dblp_acm_sim.csv']))

    alfas_t = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    output = []

    for alfa in alfas_t:
        print("============> " + str(alfa))
        # x.to_csv('x.csv',index=False)
        # x = pd.read_csv('x.csv')

        # para utilizar  a simulaçao
        # x = simulated_sbf_protocol(similarity,gold_standard_dblp_acm,alfa)
        x = pd.read_csv(getbase_dir(['results', 'sbf_03', 'intermediate_dblp_acm_'+str(alfa)+'.csv']))
        l = calculate_precision_recall_precomputed(x,gold_standard_dblp_acm,'dblp_acm',alfa)

        r = {'ds':'acm_dblp','threshold_alfa':alfa,'sbf_precision':l[0][0],
             'sbf_recall':l[0][1],'bf_precision':l[1][0],'bf_recall':l[1][1]}

        output.append(r)

    final = pd.DataFrame(output)
    final.to_csv(getbase_dir(['results', 'sbf_03', 'dblp_acm_final_result.csv']))

    data = final.copy()
    del data['ds']

    pf = data[['bf_precision','sbf_precision','threshold_alfa']].melt('threshold_alfa', var_name='bf_type', value_name='vals')
    plot_line_metric(pf,'Precision')

    pf = data[['bf_recall', 'sbf_recall','threshold_alfa']].melt('threshold_alfa', var_name='bf_type',value_name='vals')
    plot_line_metric(pf, 'Recall')

    data['bf_f1'] = 2 * ( (data['bf_precision'] * data['bf_recall']) / (data['bf_precision'] + data['bf_recall']) )
    data['sbf_f1'] = 2 * ((data['sbf_precision'] * data['sbf_recall']) / (data['sbf_precision'] + data['sbf_recall']))

    pf = data[['bf_f1', 'sbf_f1', 'threshold_alfa']].melt('threshold_alfa', var_name='bf_type',
                                                                  value_name='vals')
    plot_line_metric(pf, 'F-measure')






                 # palette="tab10", linewidth=2.5)
    plt.show()


