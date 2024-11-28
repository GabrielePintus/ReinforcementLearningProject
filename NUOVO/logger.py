import pandas as pd

class Logger:

    def __init__(self):
        self.params_evolution = pd.DataFrame(columns=['Time', 'Episode', 'Inventary', 'Bankroll', 'Loss', 'Reward'])
        self.orders_evolution = []
        self.transactions = []


    def track_params(self, **params):
        self.params_evolution = self.params_evolution.append(params, ignore_index=True)

    def track_orders(self, orders):
        self.orders_evolution.append(orders)

    def track_transactions(self, transactions):
        self.transactions.extend(transactions)

    def save(self, path):
        # Save Transactions
        df_transactions = pd.DataFrame(columns=['Time', 'Order ID', 'Transaction Volume', 'Level'])
        for t in self.transactions:
            df_transactions = df_transactions.append({
                'Time': t.time,
                'Order ID': t.order.id,
                'Transaction Volume': t.transaction_volume,
                'Level': t.level
            }, ignore_index=True)
        df_transactions.to_csv(f'transactions.csv', index=False)

        # Save Orders
        df_orders = pd.DataFrame(columns=['Time', 'Order ID', 'Price', 'Volume', 'Side', 'Market'])
        for o in self.orders_evolution:
            df_orders = df_orders.append({
                'Time': o.t,
                'Order ID': o.id,
                'Price': o.price,
                'Volume': o.volume,
                'Side': o.side,
                'Market': o.market
            }, ignore_index=True)
        df_orders.to_csv(f'orders.csv', index=False)

        # Save Params
        self.params_evolution.to_csv(f'params_evolution.csv', index=False)
