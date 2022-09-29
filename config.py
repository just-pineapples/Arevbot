SUBSCRIPTION_KEY = "fa6da66b31e94018abac3eff8316027a"
CUSTOMER_ID = "2096"
BASE = 'https://api.powerfactorscorp.com/drive/v2'

def mapper(a,b):
    list_compile = lambda a,b: a + '-' +b
    return list(map(list_compile, a,b))

def inv_reconfiure(df):
    temp_inverter = []
    new_inverter = []
    ident = [id for site, type, id, name in df.columns.values]
    
    return 
    

   