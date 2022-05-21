import pandas
import matplotlib.pyplot as pyplot
import scipy.stats as stats


class Lab4Extra:
    filename = ""
    frame = []

    def __init__(self, filename):
        self.filename = filename

    def load_and_fix(self):
        print("*" * 60)
        print(f"Загружаємо файл в фрейм.\n")
        self.frame = pandas.read_csv(self.filename + '.csv', sep=';', encoding='windows-1251')
        self.fix()
        self.show()

    def show(self):
        print(f"Дані data frame з виправленням.\n")
        self.frame.info()
        print(self.frame.describe())

    def fix(self):
        self.frame['GDP per capita'] = self.frame['GDP per capita'].str.replace(',', '.').astype(float)
        self.frame['GDP per capita'].fillna(self.frame['GDP per capita'].notna().mean(), inplace=True)

        tmp = self.frame[self.frame['GDP per capita'] < 0]
        tmp['GDP per capita'] *= -1
        self.frame[self.frame['GDP per capita'] < 0] = tmp

        self.frame['Populatiion'].fillna(self.frame['Populatiion'].notna().mean(), inplace=True)

        tmp = self.frame[self.frame['Populatiion'] < 0]
        tmp['Populatiion'] *= -1
        self.frame[self.frame['Populatiion'] < 0] = tmp

        self.frame['CO2 emission'] = self.frame['CO2 emission'].str.replace(',', '.').astype(float)
        self.frame['CO2 emission'].fillna(self.frame['CO2 emission'].notna().mean(), inplace=True)

        tmp = self.frame[self.frame['CO2 emission'] < 0]
        tmp['CO2 emission'] *= -1
        self.frame[self.frame['CO2 emission'] < 0] = tmp

        self.frame['Area'] = self.frame['Area'].str.replace(',', '.').astype(float)

        tmp = self.frame[self.frame['Area'] < 0]
        tmp['Area'] *= -1
        self.frame[self.frame['Area'] < 0] = tmp

        self.frame.to_csv(self.filename + "fixed.csv", index=False, sep=';')

    def histograms(self):
        figure, axis = pyplot.subplots(1, 4, figsize=(16, 4))
        figure.suptitle('Гістограми', fontsize=18)
        axis[0].set_title('GDP per capita')
        axis[0].hist(self.frame['GDP per capita'])
        axis[1].set_title('Population')
        axis[1].hist(self.frame['Populatiion'])
        axis[2].set_title('CO2 emission')
        axis[2].hist(self.frame['CO2 emission'])
        axis[3].set_title('Area')
        axis[3].hist(self.frame['Area'])
        pyplot.show()

    def test(self, criterion=(), alpha=0.05, sample=[]):
        stat, p = criterion(sample)
        if p > alpha:
            print(f'{criterion.__name__}(), alpha={alpha} :\t\tstat = {round(stat, 3)}, \tp = {round(p, 3)}. '
                  f'Дані відповідають нормальному розподілу. H0 не відкидається')
        else:
            print(
                f'{criterion.__name__}(), alpha={alpha} :\t\tstat = {round(stat, 3)}, \tp = {round(p, 3)}. '
                f'Дані не відповідають нормальному розподілу. H0 відкидається')
        return (sample, stat, p)

    def are_normal(self):
        print("*" * 60)
        print(f"Визначаємо параметри, що розподілені за нормальним законом")
        print("*" * 3)
        print(f"Будуємо гістограми ...")
        self.histograms()
        print(
            f"Візуально дані є нормально розподілені. Але застосуємо 3 критерії перевірки відповідності нормальному розподілу")
        self.test(criterion=stats.shapiro, sample=self.frame['GDP per capita'])
        self.test(criterion=stats.shapiro, sample=self.frame['Populatiion'])
        self.test(criterion=stats.shapiro, sample=self.frame['CO2 emission'])
        self.test(criterion=stats.shapiro, sample=self.frame['Area'])

        self.test(criterion=stats.skewtest, sample=self.frame['GDP per capita'])
        self.test(criterion=stats.skewtest, sample=self.frame['Populatiion'])
        self.test(criterion=stats.skewtest, sample=self.frame['CO2 emission'])
        self.test(criterion=stats.skewtest, sample=self.frame['Area'])

        self.test(criterion=stats.normaltest, sample=self.frame['GDP per capita'])
        self.test(criterion=stats.normaltest, sample=self.frame['Populatiion'])
        self.test(criterion=stats.normaltest, sample=self.frame['CO2 emission'])
        self.test(criterion=stats.normaltest, sample=self.frame['Area'])

    def co2_analizing(self):
        print("*" * 60)
        print(f"Шукаємо, в якому регіоні розподіл викидів СО2 найбільш близький до нормального")
        print(f"Будуємо гістограми ...")
        self.frame['CO2 emission'].hist(by=self.frame['Region'], layout=(4, 2), figsize=(10, 20))
        pyplot.show()
        print(f"Визначаємо по критеріям ...")
        for region in pandas.unique(self.frame['Region']):
            print(f"{'*' * 3}\tДля регіона {region}")
            try:
                self.test(criterion=stats.shapiro, sample=self.frame[self.frame['Region'] == region]['CO2 emission'])
            except ValueError as e:
                print(str(e))
            try:
                self.test(criterion=stats.skewtest, sample=self.frame[self.frame['Region'] == region]['CO2 emission'])
            except ValueError as e:
                print(str(e))
            try:
                self.test(criterion=stats.normaltest, sample=self.frame[self.frame['Region'] == region]['CO2 emission'])
            except ValueError as e:
                print(str(e))
        print(f"North America є нормально розподілений (по критерію Shapiro-Wilk). \n"
              f"За іншими критеріями не вдалося провірити регіон North America, так як недостатньо даних\n"
              f"Всі інші регіони не є нормально розподілені")

    def pie_chart(self):
        figure, ax = pyplot.subplots(figsize=(8, 8))
        labels = pandas.unique(self.frame['Region'])
        wedges, texts, autotexts = ax.pie(self.frame.groupby('Region').sum()['Populatiion'], labels=labels,
                                          autopct='%1.1f%%', textprops=dict(color='w'))
        ax.set_title('Населення по регіонам')
        ax.legend(wedges, labels, title='Регіони', loc='center left', bbox_to_anchor=(1, 0, 0, 1))
        pyplot.setp(autotexts, size=10, weight='bold')
        pyplot.show()


if __name__ == '__main__':
    app = Lab4Extra('Data2')
    app.load_and_fix()
    app.are_normal()
    app.co2_analizing()
    app.pie_chart()
