import time
import statistics

N = 10_000_000

def bench(fn, repeat=5):
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return times

# 1) 이터레이터 2개 - zip
def two_iters():
    a = range(N)
    b = range(N)
    s = 0
    for x, y in zip(a, b):
        s += x + y
    return s

# 2) 이터레이터 1개 - 같은 접근량 (x*2)
def one_iter():
    a = range(N)
    s = 0
    for x in a:
        s += x + x
    return s

t2 = bench(two_iters)
t1 = bench(one_iter)

print(f'이터레이터 2개 (zip): avg={statistics.mean(t2)*1000:.1f}ms  stdev={statistics.stdev(t2)*1000:.1f}ms')
print(f'이터레이터 1개      : avg={statistics.mean(t1)*1000:.1f}ms  stdev={statistics.stdev(t1)*1000:.1f}ms')
print(f'비율 (2개/1개)      : {statistics.mean(t2)/statistics.mean(t1):.3f}x')
