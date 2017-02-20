
def write_predictions_to_file(file_name, predictions, ordering=True):
    file = open(file_name, 'w')
    file.write('id,label\n')
    for i in xrange(len(predictions)):
        if ordering:
            file.write('%d,%.4f\n' % (i + 1, predictions[i]))
        else:
            file.write('%.4f\n' % predictions[i])

    file.close()
