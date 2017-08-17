import random
from PIL import Image, ImageFilter
import numpy
from scipy.ndimage import gaussian_filter
from deap import algorithms, base, creator, tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

W, H = 40, 30

def evalOneMax(individual):
    return (sum(individual),)

original = numpy.array(Image.open('images/bw.png').convert('L'))/255

def evalOneMin(individual):
  # print(individual[0])
  # print(individual.shape, original.shape)
  # a = numpy.sum(gaussian_filter(individual.reshape([H,W]), sigma=2) - original)
  # print(a)
  return (numpy.sum(individual),)

def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2

def cxTwoLineCopy(ind1, ind2):
    size = ind1.shape
    return ind1, ind2

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=H*W)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalOneMin)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def showBW(imArray):
  print(imArray)
  im = Image.new("L", [W,H])
  im.putdata(numpy.reshape(numpy.uint8((imArray)), [H*W, 1]))
  im.show()

showBW(original*255)

if __name__ == "__main__":
    pop = toolbox.population(n=500)
    
    ngen, cxpb, mutpb = 5000, 0.5, 0.2
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(ngen):
        pop = toolbox.select(pop, k=len(pop))
        pop = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
        
        invalids = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalids)
        for ind, fit in zip(invalids, fitnesses):
            ind.fitness.values = fit
    
    showBW((tools.selBest(pop, k=1)[0] + (-1))*(-255))