import numpy
import scipy.stats as stats
import random


class Lab4:

    def test(self, groups=0, count_per_group=0, criterion=(), alpha=0.05, sample=[]):
        lsample = sample
        for i in range(groups):
            lsample = numpy.concatenate(
                (
                    sample,
                    numpy.random.default_rng().normal(
                        loc=random.randint(1, groups),
                        scale=random.randint(1, groups),
                        size=count_per_group)))
        stat, p = criterion(lsample)
        if p > alpha:
            print(f'{criterion.__name__}(), {groups}x{count_per_group}, alpha={alpha} :\t\tstat = {round(stat, 3)}, \tp = {round(p, 3)}. '
                  f'Дані відповідають нормальному розподілу. H0 не відкидається')
        else:
            print(
                f'{criterion.__name__}(), {groups}x{count_per_group}, alpha={alpha} :\t\tstat = {round(stat, 3)}, \tp = {round(p, 3)}. '
                f'Дані не відповідають нормальному розподілу. H0 відкидається')
        return (lsample, stat, p)


if __name__ == '__main__':
    app = Lab4()
    print("*" * 60)
    s, stat, p1 = app.test(groups=1, count_per_group=120, criterion=stats.shapiro)  # Perform the Shapiro-Wilk test for normality.
    s, stat, p2 = app.test(groups=1, count_per_group=120, criterion=stats.skewtest,   sample=s)  # Test whether the skew is different from the normal distribution.
    s, stat, p3 = app.test(groups=1, count_per_group=120, criterion=stats.normaltest, sample=s)  # Test whether a sample differs from a normal distribution.
    if max(p1, p2, p3)==p3:
        print("Найкращою оцінкою є по критерію, який базується на тесті D’Agostino - Pearson, який поєдную в собі перекос і ексцесу")
    elif max(p1, p2, p3)==p2:
        print("Найкращою оцінкою є по критерію перекоса по лінії симетрії")
    elif max(p1, p2, p3)==p1:
        print("Найкращою оцінкою є по критерію Шапіро-Вілка")
    print("*"*60)
    print("Залежність критеріїв від значущості alpha:")
    print("*" * 1)
    app.test(groups=1, count_per_group=120, criterion=stats.shapiro, sample=s, alpha=0.025)
    app.test(groups=1, count_per_group=120, criterion=stats.shapiro, sample=s, alpha=0.05)
    app.test(groups=1, count_per_group=120, criterion=stats.shapiro, sample=s, alpha=0.1)
    app.test(groups=1, count_per_group=120, criterion=stats.shapiro, sample=s, alpha=0.2)
    app.test(groups=1, count_per_group=120, criterion=stats.shapiro, sample=s, alpha=0.3)
    app.test(groups=1, count_per_group=120, criterion=stats.shapiro, sample=s, alpha=0.4)
    print("*" * 1)
    app.test(groups=1, count_per_group=120, criterion=stats.skewtest, sample=s, alpha=0.025)
    app.test(groups=1, count_per_group=120, criterion=stats.skewtest, sample=s, alpha=0.05)
    app.test(groups=1, count_per_group=120, criterion=stats.skewtest, sample=s, alpha=0.1)
    app.test(groups=1, count_per_group=120, criterion=stats.skewtest, sample=s, alpha=0.2)
    app.test(groups=1, count_per_group=120, criterion=stats.skewtest, sample=s, alpha=0.3)
    app.test(groups=1, count_per_group=120, criterion=stats.skewtest, sample=s, alpha=0.4)
    print("*" * 1)
    app.test(groups=1, count_per_group=120, criterion=stats.normaltest, sample=s, alpha=0.025)
    app.test(groups=1, count_per_group=120, criterion=stats.normaltest, sample=s, alpha=0.05)
    app.test(groups=1, count_per_group=120, criterion=stats.normaltest, sample=s, alpha=0.1)
    app.test(groups=1, count_per_group=120, criterion=stats.normaltest, sample=s, alpha=0.2)
    app.test(groups=1, count_per_group=120, criterion=stats.normaltest, sample=s, alpha=0.3)
    app.test(groups=1, count_per_group=120, criterion=stats.normaltest, sample=s, alpha=0.4)
    print("*" * 3)
    print("При різних значеннях рівня значимості нулова гіпотеза може бути як відкинута, так і прийнята.")
    print("*" * 60)
    print("Залежність критеріїв від об'єму вибірки:")
    print("*" * 3)
    groups = 2
    count_per_group = 50
    s, _, _ = app.test(groups=groups, count_per_group=count_per_group, criterion=stats.shapiro)
    app.test(groups=groups, count_per_group=count_per_group, criterion=stats.skewtest, sample=s)
    app.test(groups=groups, count_per_group=count_per_group, criterion=stats.normaltest, sample=s)
    print("*" * 3)
    groups = 2
    count_per_group = 200
    s, _, _ = app.test(groups=groups, count_per_group=count_per_group, criterion=stats.shapiro)
    app.test(groups=groups, count_per_group=count_per_group, criterion=stats.skewtest, sample=s)
    app.test(groups=groups, count_per_group=count_per_group, criterion=stats.normaltest, sample=s)
    print("*" * 3)
    print("При різних значеннях об'єму вибірки нулова гіпотеза може бути як відкинута, так і прийнята.")
    print("*" * 60)
    print("Залежність критеріїв від кількості інтервалів групування:")
    print("*" * 3)
    groups = 2
    count_per_group = 200
    s, _, _ = app.test(groups=groups, count_per_group=count_per_group, criterion=stats.shapiro)
    app.test(groups=groups, count_per_group=count_per_group, criterion=stats.skewtest, sample=s)
    app.test(groups=groups, count_per_group=count_per_group, criterion=stats.normaltest, sample=s)
    print("*" * 3)
    groups = 5
    count_per_group = 200
    s, _, _ = app.test(groups=groups, count_per_group=count_per_group, criterion=stats.shapiro)
    app.test(groups=groups, count_per_group=count_per_group, criterion=stats.skewtest, sample=s)
    app.test(groups=groups, count_per_group=count_per_group, criterion=stats.normaltest, sample=s)
    print("*" * 3)
    print("При різних значеннях об'єму вибірки нулова гіпотеза може бути як відкинута, так і прийнята.")