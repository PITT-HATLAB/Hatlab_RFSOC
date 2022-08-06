import json
import numpy as np
from matplotlib import pyplot as plt
from Hatlab_DataProcessing.analyzer import qubit_functions_rot as qfr
from qubitMSMT.exampleConfig import config, rotResult, dataPath, sampleName

def loadData(fileName, filePath):
    with open(filePath+fileName, 'r') as infile:
        dataDict = json.load(infile)
        for k, v in dataDict.items():
            dataDict[k] = np.array(v)
    return dataDict

if __name__ == "__main__":
    dataPath = r"L:\Data\WISPE\LL_WISPE\s6\cooldown_20220401\\"
    sampleName = "LL_Wispe_0322_wispe1"

    # process pi;pulse result
    # get  piResult.Erot_result
steps = 25
freqR = 4952.8

t1_time_array = np.zeros(steps)
t1_result_array = np.zeros(steps)
t1_error_array = np.zeros(steps)

t2r_time_array = np.zeros(steps)
t2r_result_array = np.zeros(steps)
t2r_error_array = np.zeros(steps)
t2r_detuning_array = np.zeros(steps)
t2r_detuningerror_array = np.zeros(steps)

t2e_time_array = np.zeros(steps)
t2e_result_array = np.zeros(steps)
t2e_error_array = np.zeros(steps)

# T1dataArray = np.zeros((steps, 400))



for i in list(range(steps)):
    t1data = loadData(sampleName + f"_t1_{i}", dataPath)
    t2rdata = loadData(sampleName + f"_t2r_{i}", dataPath)
    t2edata = loadData(sampleName + f"_t2e_{i}", dataPath)

    t1Decay = qfr.T1Decay(t1data["x_pts_t1"], t1data["i_data"]+1j*t1data["q_data"])
    t1Result = t1Decay.run(rotResult)
    t1_result_array[i] = t1Result.lmfit_result.params["tau"].value
    t1_error_array[i] = t1Result.lmfit_result.params["tau"].stderr
    t1_time_array[i] = t1data["t"]/60
    # t1Result.plot(num="1")
    # T1dataArray[i] = t1Result.lmfit_result.data

    t2Ramsey = qfr.T2Ramsey(t2rdata["x_pts_t2r"], t2rdata["i_data"]+1j*t2rdata["q_data"])
    t2rResult = t2Ramsey.run(rotResult)
    t2r_result_array[i] = t2rResult.lmfit_result.params["tau"].value
    t2r_error_array[i] = t2rResult.lmfit_result.params["tau"].stderr
    t2r_detuning_array[i] = (t2rResult.lmfit_result.params["f"].value)*1e6+ freqR
    t2r_detuningerror_array[i] = (t2rResult.lmfit_result.params["f"].stderr)*1e6
    t2r_time_array[i] = t2rdata["t"]/60

    t2EDecay = qfr.T1Decay(t2edata["x_pts_T2E"], t2edata["i_data"]+1j*t2edata["q_data"])
    t2EResult = t2EDecay.run(rotResult)
    t2e_result_array[i] = t2EResult.lmfit_result.params["tau"].value
    t2e_error_array[i] = t2EResult.lmfit_result.params["tau"].stderr
    t2e_time_array[i] = t2edata["t"]/60
    # plt.figure()
    # plt.plot(t2rResult.lmfit_result.data)



plt.figure(figsize=(5,4))
plt.subplot(111,title=sampleName + f" T1 over time", xlabel="Time (Minutes)", ylabel="T1 (us)")
plt.errorbar(t1_time_array, t1_result_array,yerr =t1_error_array, markersize = 1)
# plt.ylim((40, 140))
plt.show()

plt.figure(figsize=(5,4))
plt.subplot(111,title=sampleName + f" Qubit Freq over time", xlabel="Time (Minutes)", ylabel="Frequency")
plt.errorbar(t1_time_array, t2r_detuning_array/1e9,yerr =t2r_detuningerror_array/1e9, markersize = 1)
# plt.ylim((40, 140))
plt.show()

plt.figure(figsize=(5,4))
plt.subplot(111,title=sampleName + f" T2R over time", xlabel="Time (Minutes)", ylabel="T2R (us)")
plt.errorbar(t2r_time_array, t2r_result_array,yerr =t2r_error_array, markersize = 1)
plt.show()

plt.figure(figsize=(5,4))
plt.subplot(111,title=sampleName + f" T2E over time", xlabel="Time (Minutes)", ylabel="T2E (us)")
plt.errorbar(t2e_time_array, t2e_result_array,yerr =t2e_error_array, markersize = 1)
plt.show()

print(np.average(t1_result_array))
print(np.std(t1_result_array))

print(np.average(t2r_detuning_array))
print(np.std(t2r_detuning_array))

print(np.average(t2r_result_array))
print(np.std(t2r_result_array))

print(np.average(t2e_result_array))
print(np.std(t2e_result_array))






