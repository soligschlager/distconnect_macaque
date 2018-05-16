def model_select(df, outcome_var, predictor_vars):

    from itertools import combinations
    import pandas as pd
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    df_select = pd.DataFrame(columns=['model', 'AIC', 'BIC', 'R2', 'F', 'P'])
    index = 0

    for n in range(len(predictor_vars)):
        for i in combinations(predictor_vars, n+1):

            formula = '%s ~' % outcome_var
            for n_p, p in enumerate(i):
                if n_p == 0: 
                    formula = formula + ' %s' %p
                else:
                    formula = formula + ' + %s' %p

            #df_ols = df_regional[vars]
            #df_ols = df_ols.dropna() # happens automatically anyway
            
            res = smf.ols(formula=formula, data=df).fit()
            df_select.loc[index] = [formula, res.aic, res.bic, res.rsquared, res.fvalue, res.f_pvalue]
            index += 1
            
    return df_select.sort_values('AIC')
