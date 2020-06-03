using Microsoft.VisualBasic.FileIO;
using System;
using System.ComponentModel.Design;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.IO.Enumeration;
using System.Linq;
using System.Reflection.Metadata;
using System.Security.Cryptography;

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



        private static UInt32[,] process_korea_format(preProccessedData raw_CSV)
        {

            UInt32[,] processed_data = new UInt32[raw_CSV.num_rows,8];
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
                    processed_data[processed_data_row, 0] = (UInt32) processed_data_row; 

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
                    processed_data[processed_data_row, 4] = UInt32.Parse(raw_CSV.data[curr_row, age_col]);

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
        const string formatter = "{0,5}{1,17}{2,15}";

        // Convert four byte array elements to a uint and display it.
        public static void ProcessBA(byte[] bytes, int index)
        {


            // Double check for UInt32 Values 


                UInt32 value = BitConverter.ToUInt32(bytes, index);
                Console.WriteLine(formatter, index, BitConverter.ToString(bytes, index, sizeof(UInt32)), value);
            

        }




        static int Main(string[] args)
        {
            bool verbose = true;
            int num_cols = 8;
            const int pad_bytes = 2; 


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
            UInt32[,] processed_data = process_korea_format(data);
            BinaryWriter file_writer = new BinaryWriter(File.Open(args[1], FileMode.Create));

            // Loop over rows
            //for (int i = 0; i <= processed_data.GetUpperBound(0); i++)
            for (int i = 0; i <= 1; i++)
            {
                if (processed_data[i,0] != 0)
                {
                    // Print processed data 
                    if (verbose)
                    {
                        for (int k = 0; k < num_cols; k++)
                            Console.Write("{0} ", processed_data[i, k]);
                        Console.WriteLine("");

                    }

                    // Store each row in bytes
                    byte[] byted_row = new byte[8 * sizeof(UInt32)];
                    for (int j = 0; j < 8; j++)
                    {
                        Buffer.BlockCopy(BitConverter.GetBytes(processed_data[i,j]), 0, byted_row, j*sizeof(UInt32), sizeof(UInt32));

                    }

                    // Randomize padded byte data 

                        for (int k = 0; k < num_cols; k++)
                        {
                        if (verbose)
                        {
                            //Console.WriteLine("Before Rand:");
                            ProcessBA(byted_row, k * sizeof(UInt32));

                        }

                        RNGCryptoServiceProvider rng = new RNGCryptoServiceProvider();
                        rng.GetBytes(byted_row, pad_bytes + k * sizeof(UInt32), 2); 
                  
                            if (verbose)
                            {
                                //Console.WriteLine("After Rand:");
                                ProcessBA(byted_row, k * sizeof(UInt32));

                            }
                    }
                    // Encrypt Bytes: 
                    var key_pair = new RSACryptoServiceProvider(128); 
                    // write row to file

                    file_writer.Write(byted_row);


                }
            }

            file_writer.Flush();
            file_writer.Close();


            // Write CSV  

            return 0; 
        }
    }
}
