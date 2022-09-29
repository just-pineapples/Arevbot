SUBSCRIPTION_KEY = "fa6da66b31e94018abac3eff8316027a"
CUSTOMER_ID = "2096"
BASE = 'https://api.powerfactorscorp.com/drive/v2'


def mapper(a,b):
    list_compile = lambda a,b: a + '-' +b
    return list(map(list_compile, a,b))


def timezone_convert(df):

    timezone = df.iat[6,0]
    if timezone == 'Eastern Standard Time':
        return 'US/Eastern'
    elif timezone == 'Central Standard Time':
        return 'US/Central'
    elif timezone == 'Mountain Standard Time':
        return 'US/Mountain'
    elif timezone == 'Pacific Standard Time':
        return 'US/Pacific'
    