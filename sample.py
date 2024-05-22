import sys
import subprocess
import multiprocessing

def run_llmSMGP(problem, run):
    file = open("logs/llmSMGP_" + problem + "_" + str(run) + ".log", "w")
    process = subprocess.Popen(["python", "llmSMGP.py", "--problem", problem, "--run", str(run)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True)
    
    for line in process.stdout:
        file.write(line)
        print(line)
    for line in process.stderr:
        file.write(line)
        print(line)

    file.close()
    process.wait()

def main():
    problems = sys.argv[1:-1]
    run = int(sys.argv[2])

    # print(type(problems))

    # problems = [int(item) for item in args.problems.split(' ')]

    pool = multiprocessing.Pool(processes=2)

    print("Problems to solve:")
    print(problems)

    results = []
    for problem in problems:
        result = pool.apply_async(run_llmSMGP, args=(problem, run))
        results.append(result)
    
    [result.wait() for result in results]
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()