import tkinter as tk
from tkinter import filedialog
from data_preprocessing import DataProcessor
from k_means import KMeans
import numpy as np

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("K-Means Clustering")

        self.file_label = tk.Label(master, text="CSV File Path:")
        self.file_label.grid(row=0, column=0)

        self.file_entry = tk.Entry(master)
        self.file_entry.grid(row=0, column=1)

        self.browse_button = tk.Button(master, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2)

        self.percentage_label = tk.Label(master, text="Percentage of Data:")
        self.percentage_label.grid(row=1, column=0)

        self.percentage_entry = tk.Entry(master)
        self.percentage_entry.grid(row=1, column=1)

        self.k_label = tk.Label(master, text="Number of Clusters :")
        self.k_label.grid(row=2, column=0)

        self.k_entry = tk.Entry(master)
        self.k_entry.grid(row=2, column=1)

        self.run_button = tk.Button(master, text="Apply K-Means Clustering", command=self.run_kmeans, width=30)
        self.run_button.grid(row=3, column=1)

        self.output_text = tk.Text(master, height=20, width=80)
        self.output_text.grid(row=4, column=0, columnspan=3)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, file_path)

    def run_kmeans(self):
        file_path = self.file_entry.get()
        percentage = float(self.percentage_entry.get())
        k = int(self.k_entry.get())

        columns_to_drop = ['Release Year', 'Metascore', 'Genre', 'Director', 'Cast', 'Gross', 'Votes']

        data_processor = DataProcessor(file_path, columns_to_drop)
        preprocessed_data, outliers_list = data_processor.preprocess_data(percentage=percentage, remove_outliers=True)

        if preprocessed_data.empty:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Error: No data found for clustering after preprocessing.")
            return

        ratings_train = np.array(preprocessed_data['Duration'])

        if not ratings_train.any():
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Error: No data found for clustering.")
            return

        movie_names = preprocessed_data['Movie Name']

        kmeans = KMeans(ratings_train, movie_names)
        kmeans.cluster_movies(k)
        clustered_movies = kmeans.get_clustered_movies()  # Store clustered movies here
        centroids_history = kmeans.iteration_centroids  # Get centroids after each iteration

        output_text = "Clusters:-\n"
        for cluster_id, movies_in_cluster in enumerate(clustered_movies):
            output_text += f"Cluster {cluster_id + 1}:\n"
            for movie_index in movies_in_cluster:
                if movie_index < len(preprocessed_data):  # Check if the index is within the valid range
                    movie_name = preprocessed_data.iloc[movie_index]['Movie Name']
                    rating = ratings_train[movie_index]
                    output_text += f"Movie Name: {movie_name}, Duration: {rating}\n"
                else:
                    print(f"Invalid index: {movie_index} for preprocessed_data.")

        output_text += "Outliers:-\n"
        for outlier_info in outliers_list:
            movie_name, rating = outlier_info
            output_text += f"Outlier Movie Name: {movie_name}, Rating: {rating}\n"

        output_text += "Centroids After Each Iteration:-\n"
        for iteration, centroids in enumerate(centroids_history):
            output_text += f"Iteration {iteration + 1}:\n"
            for i, centroid in enumerate(centroids):
                output_text += f"Centroid {i + 1}: {centroid}\n"



        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, output_text)
