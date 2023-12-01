import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, data):
        self.data = data
        self.balance = 100000  # Starter med 100.000
        self.in_trade = False
        self.buy_price = 0
        self.trade_amount = 0  # Beløp investert i den nåværende handelen
        self.trade_history = []

    def trade(self):
        for index, row in self.data.iterrows():
            trade_limit = self.balance * 0.02  # 2% av balansen

            if self.in_trade:
                # For profitability
                profit_or_loss_percent = (row['Close'] - self.buy_price) / self.buy_price

                # If there's a sell signal and profit is greater than 0.3%
                if row['Filtered Prediction'] == 'Sell' and profit_or_loss_percent >= 0.002:
                    profit_or_loss = self.trade_amount * profit_or_loss_percent
                    self.balance += self.trade_amount + profit_or_loss  # legger tilbake handelsbeløpet og profitten
                    print(f"Sold at {index} for {row['Close']}. Profit: {profit_or_loss}. Balance: {self.balance}")
                    self.in_trade = False
                    self.trade_history.append({
                        'type': 'sell',
                        'price': row['Close'],
                        'reason': 'profit',
                        'profit_or_loss': profit_or_loss
                    })

                # If the loss is greater than 0.3%
                elif profit_or_loss_percent <= -0.004:
                    profit_or_loss = self.trade_amount * profit_or_loss_percent
                    self.balance += self.trade_amount + profit_or_loss
                    print(f"Sold at {index} due to loss for {row['Close']}. Loss: {profit_or_loss}. Balance: {self.balance}")
                    self.in_trade = False
                    self.trade_history.append({
                        'type': 'sell',
                        'price': row['Close'],
                        'reason': 'loss',
                        'profit_or_loss': profit_or_loss
                    })

            # If there's a buy signal and not in a trade
            if not self.in_trade and row['Filtered Prediction'] == 'Buy':
                self.buy_price = row['Close']
                self.trade_amount = trade_limit  # bruker 2% av balansen for denne handelen
                self.balance -= self.trade_amount  # trekker fra handelsbeløpet fra balansen
                print(f"Bought at {index} for {row['Close']} using amount: {self.trade_amount}. Balance: {self.balance}")
                self.in_trade = True
                self.trade_history.append({
                    'type': 'buy',
                    'price': row['Close'],
                    'amount': self.trade_amount
                })

        return self.balance

    def plot_prices_with_signals(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data.index, self.data['Close'], label='Close Price', alpha=0.5)

        # Marker kjøpssignaler med grønne triangler
        plt.scatter(self.data[self.data['Filtered Prediction'] == 'Buy'].index,
                    self.data[self.data['Filtered Prediction'] == 'Buy']['Close'],
                    label='Buy Signal', marker='^', alpha=1, color='g')

        # Marker salgssignaler med røde triangler
        plt.scatter(self.data[self.data['Filtered Prediction'] == 'Sell'].index,
                    self.data[self.data['Filtered Prediction'] == 'Sell']['Close'],
                    label='Sell Signal', marker='v', alpha=1, color='r')

        plt.title('Close Price & Buy/Sell Signals')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()

    def plot_net_profit_over_time(self):
        cumulative_profit = [t['profit_or_loss'] for t in self.trade_history if 'profit_or_loss' in t]
        for i in range(1, len(cumulative_profit)):
            cumulative_profit[i] += cumulative_profit[i-1]

        plt.figure(figsize=(14, 7))
        plt.plot(range(len(cumulative_profit)), cumulative_profit)
        plt.title('Net Profit over Time')
        plt.xlabel('Trade Number')
        plt.ylabel('Net Profit')
        plt.show()

    def plot_profit_distribution(self):
        profits = [t['profit_or_loss'] for t in self.trade_history if 'profit_or_loss' in t]

        plt.figure(figsize=(14, 7))
        plt.hist(profits, bins=50, alpha=0.75)
        plt.title('Profit/Loss per Trade Distribution')
        plt.xlabel('Profit/Loss')
        plt.ylabel('Number of Trades')
        plt.show()

    def plot_trade_outcomes(self):
        successful = len([t for t in self.trade_history if 'reason' in t and t['reason'] == 'profit'])
        failed = len(self.trade_history) - successful

        plt.figure(figsize=(7, 7))
        plt.pie([successful, failed], labels=['Successful', 'Failed'], autopct='%1.1f%%', startangle=90, colors=['g', 'r'])
        plt.title('Trade Outcomes')
        plt.show()
