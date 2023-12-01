from Binance.tickerHandlere import BinanceData
from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import pytz
from pytz import UTC
from tqdm import tqdm
import os  

def download_and_save_data(csv_file_name, start):  
    from Binance.tickerHandlere import BinanceData  
    from binance.client import Client  
    from datetime import datetime, timedelta  
    import pandas as pd  
    import pytz  
    from pytz import UTC  
    from tqdm import tqdm  
    import os  
  
    apikey = "8aonjjjJ0EJt1coxcR335wl1Qsq4eYov9k2CuXUxP7IQhiBwDPOrbp5D6LANdbkW"  
    apisecret = "Te0MrdyCGhTcNGvmTvWDfSEG4KLPRlaIecaC0PqTtiCAnO2bB5T8miWsfQJUYDSW"  
    binanceData = BinanceData(apikey, apisecret)  
  
    # Start- og sluttdato  
   
    end = datetime.now(pytz.UTC)  
    end = end - timedelta(hours=1)  
  
    all_data_list = []  
    interval = timedelta(days=30)  
  
    # For å beregne totalt antall dager for tqdm  
    total_days = (end - start).days  
  
    with tqdm(total=total_days) as pbar:  
        current_start = start  
        while current_start < end:  
            current_end = min(current_start + interval, end)  
            print(current_end)  
            # Hente data for det nåværende intervallet  
            try:  
                data = binanceData.get_historical_klines(  
                    "BTCUSDT",  
                    Client.KLINE_INTERVAL_1HOUR,  
                    current_start.strftime("%d %b, %Y %H:%M UTC"),  
                    current_end.strftime("%d %b, %Y %H:%M UTC")  
                )  
                if not data.empty:  
                    all_data_list.append(data)  
                    new_start = data.index[-1].replace(tzinfo=UTC)  
  
                    # Sjekk for å unngå en endeløs løkke  
                    if new_start == current_start:  
                        current_start += interval  
                    else:  
                        current_start = new_start  
                else:  
                    current_start = current_end  
            except Exception as e:  
                print(f"Feil under henting av data: {e}")  
                break  
  
            # Oppdaterer tqdm fremdriftslinjen  
            days_retrieved = (current_end - current_start).days  
            pbar.update(days_retrieved)  
  
    all_data = pd.concat(all_data_list, ignore_index=False)  
  
    # Finn gjeldende katalog  
    current_directory = os.path.dirname(os.path.abspath(__file__))  
  
    # Opprett underkatalogen 'CSV' hvis den ikke eksisterer  
    csv_directory = os.path.join(current_directory, 'CSV')  
    if not os.path.exists(csv_directory):  
        os.makedirs(csv_directory)  
  
    # Lagre filen i 'CSV'-katalogen  
    file_path = os.path.join(csv_directory, f'{csv_file_name}.csv')  
    all_data.to_csv(file_path, mode='w', index=True)  
  
# Bruk funksjonen med filnavnet som parameter  
download_and_save_data('onehour_2023-06-01-now', datetime(2023, 6, 1, tzinfo=UTC)  )  
