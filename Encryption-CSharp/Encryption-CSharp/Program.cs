using Microsoft.VisualBasic.FileIO;
using System;
using System.IO.Enumeration;
using System.Linq;

namespace RSACryption
{
    public enum RSAKeySize
    {
        Key_64 = 64, 
        Key_128 = 128
    }
}

namespace Encryption_CSharp
{
    class Encryption_Module
    {
        public static int id_col = 0, 
                          gender_col = 2, 
                          age_col = 3 , 
                          state_col = 17; 
        public struct preProccessedData
        {
            public int num_rows;
            public int num_cols;
            public string[,] data; 
        }
        
        private static preProccessedData CSV_Loader(string file)
        {
            string all_file_data = System.IO.File.ReadAllText(file);

            // Split up CSV file 
            all_file_data = all_file_data.Replace('\n', '\r');
            string[] rows = all_file_data.Split(new char[] { '\r' }, StringSplitOptions.RemoveEmptyEntries);

            // Find CSV properties 
            int num_rows = rows.Length;
            int num_cols = rows[0].Split(',').Length;

            // Load into struct 
            preProccessedData loaded_data; 
            loaded_data.data = new string[num_rows, num_cols];
            loaded_data.num_rows = num_rows;
            loaded_data.num_cols = num_cols; 


            // Copy over cells 
            for (int i = 0; i < num_rows; i++)
            {
                string[] a_row = rows[i].Split(',');
                for (int j = 0; j < num_cols; j++)
                {
                    loaded_data.data[i, j] = a_row[j]; 
                }
            }

            return loaded_data; 
        }

        private static Int64[,] process_korea_format(preProccessedData raw_CSV)
        {

            Int64[,] processed_data = new Int64[raw_CSV.num_rows,8];
            int processed_data_row = 0; 

            // Loop over all rows in raw_csv / preprocessed data 
            for (int curr_row = 1; curr_row < raw_CSV.num_rows; curr_row++)
            {
                // Check if the row has data we want 
                if(raw_CSV.data[curr_row, id_col] == "" || raw_CSV.data[curr_row, gender_col] == "" || raw_CSV.data[curr_row, age_col] == "" || raw_CSV.data[curr_row, state_col] == "")
                {
                    continue; 
                }
                else
                {
                    // Copy over patient ID
                    processed_data[processed_data_row, 0] = Int64.Parse(raw_CSV.data[curr_row, id_col]);

                    // Copy over gender data 
                    if (raw_CSV.data[curr_row, gender_col] == "male")
                    {
                        processed_data[processed_data_row, 1] = 1;
                        processed_data[processed_data_row, 2] = 0;
                        processed_data[processed_data_row, 3] = 0;

                    }
                    else if (raw_CSV.data[curr_row, gender_col] == "female")
                    {
                        processed_data[processed_data_row, 1] = 0;
                        processed_data[processed_data_row, 2] = 1;
                        processed_data[processed_data_row, 3] = 0;
                    }
                    else
                    {
                        processed_data[processed_data_row, 1] = 0;
                        processed_data[processed_data_row, 2] = 0;
                        processed_data[processed_data_row, 3] = 1;
                    }

                    // Copy over age
                    processed_data[processed_data_row, 4] = Int64.Parse(raw_CSV.data[curr_row, age_col]);

                    // Copy over status 

                    // Copy over gender data 
                    if (raw_CSV.data[curr_row, state_col] == "deceased")
                    {
                        processed_data[processed_data_row, 5] = 1;
                        processed_data[processed_data_row, 6] = 0;
                        processed_data[processed_data_row, 7] = 0;

                    }
                    else if (raw_CSV.data[curr_row, state_col] == "released")
                    {
                        processed_data[processed_data_row, 5] = 0;
                        processed_data[processed_data_row, 6] = 1;
                        processed_data[processed_data_row, 7] = 0;
                    }
                    else
                    {
                        processed_data[processed_data_row, 5] = 0;
                        processed_data[processed_data_row, 6] = 0;
                        processed_data[processed_data_row, 7] = 1;
                    }

                    // Move to next row in processed data
                    processed_data_row++; 
                }

            }
            return processed_data; 

        }

        static int Main(string[] args)
        {

            // Validate inputs 
            if (args.Length == 0)
            {
                Console.WriteLine("Usage: encrypt.exe [INPUT CSV DATA] [OUTPUT CSV] [OUTPUT PRIVATE KEY] [OUTPUT PUBLIC KEY]");
                return 1; 
            }

            // Load file locations 
            String input_csv = args[0];
            // String output_csv = args[1];
            // String output_private_key = args[2];
            // String output_public_key = args[3];

            // Load CSV into 2D Array 
            preProccessedData data = CSV_Loader(input_csv);

            // Process Data 
            Int64[,] processed_data = process_korea_format(data);

            Console.WriteLine(String.Join(" ", processed_data.Cast<Int64>()));
            // Encrypt CSV Array 



            // Write CSV Array 

            return 0; 
        }
    }
}
