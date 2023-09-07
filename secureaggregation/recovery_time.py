import sys
def recovery_time_num(compute_time,num,dropnum,index):
            with open("./recovery_time_num.txt",'a+') as f:
                if index==0:
                    f.write('total num:{},dropout num:{}\n'.format(num,dropnum))
                    f.write('{}\n'.format(compute_time))
                else:
                    f.write('{}\n'.format(compute_time))

def non_dropout_aggregate_time_num(compute_time,num,dropnum,index):
            with open("./non_dropout_aggregation_time_num.txt",'a+') as f:
                if index==0:
                    f.write('total num:{},dropout num:{}\n'.format(num,dropnum))
                    f.write('{}\n'.format(compute_time))
                else:
                    f.write('{}\n'.format(compute_time))

def dropout_aggregation_time_num(compute_time,num,dropnum,index):
            with open("./dropout_aggregation_time_num.txt",'a+') as f:
                if index==0:
                    f.write('total num:{},dropout num:{}\n'.format(num,dropnum))
                    f.write('{}\n'.format(compute_time))
                else:
                    f.write('{}\n'.format(compute_time))

def recovery_time_gradient(compute_time,shape,index):
            with open("./recovery_time_gradients.txt",'a+') as f:
                if index==0:
                    f.write('dimension:{}\n'.format(shape))
                    f.write('{}\n'.format(compute_time))
                else:
                    f.write('{}\n'.format(compute_time))

def nondropout_aggregation_time_gradient(compute_time,shape,index):
            with open("./nondropout_aggregation_time_gradients.txt",'a+') as f:
                if index==0:
                    f.write('dimension:{}\n'.format(shape))
                    f.write('{}\n'.format(compute_time))
                else:
                    f.write('{}\n'.format(compute_time))

def dropout_aggregation_time_gradient(compute_time,shape,index):
            with open("./dropout_aggregation_gradients.txt",'a+') as f:
                if index==0:
                    f.write('dimension:{}\n'.format(shape))
                    f.write('{}\n'.format(compute_time))
                else:
                    f.write('{}\n'.format(compute_time))

def recovery_cpu_num(compute_time,num,dropnum,index):
            with open("./recovery_cpu_num.txt",'a+') as f:
                if index==0:
                    f.write('total num:{},dropout num:{}\n'.format(num,dropnum))
                    f.write('{}\n'.format(compute_time))
                else:
                    f.write('{}\n'.format(compute_time))

def recovery_cpu_gradient(compute_time,shape,index):
            with open("./recovery_cpu_gradients.txt",'a+') as f:
                if index==0:
                    f.write('dimension:{}\n'.format(shape))
                    f.write('{}\n'.format(compute_time))
                else:
                    f.write('{}\n'.format(compute_time))

def compute_size(dict):
      size=0
      for key,values in dict.items():
            for i in range(len(values)):
                  size += sys.getsizeof(values[i])
      return size/1024