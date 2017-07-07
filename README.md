# future
# depend
	pip install pandas
# bash 
	alias="ls -l"
	PATH=$PATH:/usr/local/cuda-8.0/bin
	LD_LIBRARY_PATH=$PATH:/usr/local/cuda-8.0/lib64
	export CUDA_HOME=/usr/local/cuda
# cuda
	wget "https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run"
# cudnn 
	tar xvzf cudnn-8.0-linux-x64-v5.1.tgz
	cp cuda/include/cudnn.h /usr/local/cuda/include
	cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
	chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

# tensorflow 
	wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp27-none-linux_x86_64.whl
	pip install tensorflow_gpu-1.0.0-cp27-none-linux_x86_64.whl

	
def cal_kappa(y, pred, nclasses=3):
    nclasses = max(y) - min(y) + 1 
    o = np.zeros([nclasses,nclasses])
    w =  np.zeros([nclasses,nclasses])
    y_hist = np.zeros(nclasses)
    pred_hist = np.zeros(nclasses)
    for i in xrange(nclasses):
        for j in xrange(nclasses):
            w[i,j] = (i-j)**2
    for i in xrange(y.shape[0]):
        o[int(round(y[i])), int(round(pred[i]))] += 1
        y_hist[int(round(y[i]))] += 1
        pred_hist[int(round(pred[i]))] += 1
    e = np.outer(y_hist, pred_hist)
    rescale = np.sum(e) / np.sum(o)
    return 1 - rescale * np.sum(o * w) / np.sum(e * w)

