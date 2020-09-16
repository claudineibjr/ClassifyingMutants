# Shapiro-Wilk Test
from scipy.stats import shapiro
import sys 

def shapiroTests(metric):
    alpha = 0.05

    if metric == 'Accuracy':
        minimalData = [73.3044272899983, 80.7125447899382, 82.124401798812, 67.2154666570473, 68.5171043455258, 68.2305629704446, 54.7904011254599] # Accuracy
        equivalentData = [77.1330804311774, 78.044315920398, 80.4182711442786, 72.8765630182421, 73.7657918739635, 73.5212064676617, 56.5719610281924] # Accuracy
    elif metric == 'Precision':
        minimalData = [73.2042222029992, 81.1591009574172, 82.0663174740615, 65.8126209751859, 68.5641799941176, 68.4417426012128, 63.9073920257286] # Precision
        equivalentData = [75.1602587318472, 77.6291653312547, 79.8367822892124, 74.9293351229949, 73.0914534742774, 73.3064707522537, 53.5734200073755] # Precision
    elif metric == 'Recall':
        minimalData = [73.6888151408027, 80.1871317605752, 82.4008970011784, 72.0026573359305, 68.5953514657432, 67.8183752975976, 41.1570834235144] # Recall
        equivalentData = [81.18092039801, 78.946144278607, 81.7907960199005, 68.8770895522388, 75.3478606965174, 74.0834411276949, 98.7735323383085] # Recall
    elif metric == 'F1':
        minimalData = [73.388811279031, 80.5751398819614, 82.2173988831673, 68.6953986307584, 68.526167547662, 68.0727754151874, 39.2478304713536] # F1-Score
        equivalentData = [78.0153028826119, 78.2515693674748, 80.745503717465, 71.7132143674435, 74.1602137731567, 73.652505438322, 69.4648175267809] # F1-Score
    else:
        sys.exit()

    print('=== {} ========='.format(metric))

    # === Minimal Mutants =========
    print('--- Minimal Mutants ---------')
    
    statMinimal, pMinimal = shapiro(minimalData)
    print('Statistics = %.3f, p=%.3f' % (statMinimal, pMinimal))

    if pMinimal > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

    # === Minimal Mutants =========
    print('\n--- Equivalent Mutants ---------')
    
    statEquivalent, pEquivalent = shapiro(equivalentData)
    print('Statistics = %.3f, p=%.3f' % (statEquivalent, pEquivalent))

    if pEquivalent > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

if __name__ == '__main__':
    shapiroTests('Accuracy')
    shapiroTests('Precision')
    shapiroTests('Recall')
    shapiroTests('F1')