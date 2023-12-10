import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # Create a log file and try different parameters for the GRU model
    with open("gru_logs.txt", 'w') as logfile:
      # Different values for our parameters
      sequence_lengths = [3,4,5]

      learning_rates = [0.0001, 0.00001, 0.000001]
      epochs = [25, 50, 100]
      hidden_sizes = [5, 10, 20]

      # Store the parameters and errors after training the GRU model
      values = []

      # The cartesian product of all possible parameters
      for sequence_length, learning_rate, num_epochs, hidden_size in itertools.product(sequence_lengths,
                                                                                      learning_rates,
                                                                                      epochs,
                                                                                      hidden_sizes):
        # Train the GRU model with the given parameters
        # We keep the sequences in a Many-One structure (input sequences have multiple values
        # while output sequence will always be of length one)
        index_data_obj = IndexData('https://github.com/nikhilmanda9/ML-Stock-Market-Prediction/blob/main/all_stocks_5yr.parquet?raw=true',
                                  sequence_length)

        gru = GRU(x_train=index_data_obj.x_train,
              y_train=index_data_obj.y_train,
              learning_rate=learning_rate,
              num_epochs=num_epochs,
              input_size=1,
              hidden_size=hidden_size,
              output_size=1,
              sequence_length=sequence_length)
        gru.fit()

        # Get the parameters, data, and errors for the model
        x_train = index_data_obj.x_train
        y_train = index_data_obj.y_train
        x_test = index_data_obj.x_test
        y_test = index_data_obj.y_test
        train_errs = gru.train_errs
        train_err = train_errs[-1]
        test_output = gru.predict(index_data_obj.x_test, gru.weights)
        test_err = np.sum(np.square((test_output - index_data_obj.y_test))) / (sequence_length * gru.num_sequences)

        values.append([sequence_length,
                       learning_rate,
                       num_epochs,
                       hidden_size,
                       train_err,
                       test_err,
                       x_train,
                       y_train,
                       x_test,
                       y_test,
                       train_errs,
                       test_output])

      # Write the primary parameters and errors to the logfile
      logfile.write(tabulate(values[:6],
                    headers=['Sequence Length', 'Learning Rate', 'Number of Epochs',
                            'Hidden Size', 'Training Error', 'Testing Error'],
                    floatfmt=["", "", "", "", ".4f", ".4f"]))

      # Extract the best parameters based on the lowest average Training and Testing Errors
      best_sequence_length, best_learning_rate, best_num_epochs, best_hidden_size = None, None, None, None
      best_train_err, best_test_err, best_avg_err = None, None, 1e9
      best_x_train, best_y_train, best_x_test, best_y_test = None, None, None, None
      best_train_errs, best_test_output = None, None
      for sequence_length, learning_rate, num_epochs, hidden_size, train_err, test_err, x_train, y_train, x_test, y_test, train_err, test_output in values:
        avg_err = (train_err + test_err) / 2
        if avg_err < best_avg_err:
          best_sequence_length = sequence_length
          best_learning_rate = learning_rate
          best_num_epochs = num_epochs
          best_hidden_size = hidden_size
          best_train_err = train_err
          best_test_err = test_err
          best_avg_err = avg_err
          best_x_train, best_y_train, best_x_test, best_y_test = x_train, y_train, x_test, y_test
          best_train_errs, best_test_output = train_err, test_output

      # Print the best parameters and errors
      print("Sequence Length: {}\nLearning Rate: {}\nNumber of Epochs: {}\nHidden Size: {}\nTraining Error: {:.4f}\nTesting Error: {:.4f}".format(
          best_sequence_length, best_learning_rate, best_num_epochs, best_hidden_size, best_train_err, best_test_err))
      
      print("\n\n")

      # Plot the errors
      x = np.arange(0, best_num_epochs)
      plt.plot(x, best_train_errs, "r")
      plt.title("Training Errors Across {} Epochs".format(best_num_epochs))
      plt.xlabel("Epochs")
      plt.ylabel("Error")

      print("\n\n")

      # Visualize the computed S&P 500 Index
      fig = plt.subplots(figsize=(16, 5))
      plt.plot(index_data_obj.index_data["date"][:1000], index_data_obj.index_data["index"][:1000], color='r')
      plt.plot(index_data_obj.index_data["date"][1000:], index_data_obj.index_data["index"][1000:], color='b')
      plt.title("Daily S&P 500 Index")
      plt.legend(['Train', 'Test'])
      plt.xlabel('Date')
      plt.ylabel('Index Price (USD)')
      plt.show()