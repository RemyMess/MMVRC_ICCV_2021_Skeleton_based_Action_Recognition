import csv

def csv_writer(predictions,file_path='my_submission.csv'):
    with open(file_path,'w+',newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['index',' category'])
        for i,p in enumerate(predictions):
            writer.writerow([i,p])
    print('Prediction written to {}.'.format(file_path))
    
