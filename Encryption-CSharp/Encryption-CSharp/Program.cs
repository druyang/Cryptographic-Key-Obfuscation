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
        // Data Constants 
        public static int id_col = 0, 
                          gender_col = 2, 
                          age_col = 3 , 
                          state_col = 17;

        // Encryption Constants 
        const UInt64 e = 963443092119039113;
        const UInt64 d = 920403722748280569;
        const UInt64 n = 2108958572404460311;


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


        // ENCRYPTION FUNCTIONS: 
        // Computes (a * b) % n
        private UInt64 modmult(UInt64 a, UInt64 b, UInt64 n)
        {
            UInt64 res = 0;
            a = a % n; 
             while (b > 0)
            {
                if (b % 2 == 1)
                    res = (res + a) % n;

                b = b >> 1;
                a = (a * 2) % n; 
            }

            return res; 

        }

        // Computes (msg ** exponent) % n
        private UInt64 modexp(UInt64 msg, UInt64 exponent, UInt64 n)
        {
            UInt64 res = 1; 
            msg = msg % n; // Update msg if it is more than or equal to n
            while (exponent > 0)
            {
                // If exponent is odd, multiply x with result
                if (exponent % 2 == 1)
                    res = modmult(res, msg, n);

                // exponent must be even now
                exponent = exponent >> 1; // exponent = exponent/2
                msg = modmult(msg, msg, n); // compute (msg^2) % n
            }

            return res;
        }

        // Actual encryption function 
        public UInt64 encryptCell(UInt64 data)
        {
            return modexp(data, e, n);
        }


        private static UInt64[,] process_korea_format(preProccessedData raw_CSV)
        {

            UInt64[,] processed_data = new UInt64[raw_CSV.num_rows,8];
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
                    processed_data[processed_data_row, 0] = (UInt64) processed_data_row; 

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
                    processed_data[processed_data_row, 4] = UInt64.Parse(raw_CSV.data[curr_row, age_col]);

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
        const string formatter = "{0,5}{1,27}{2,24}";

        // Convert four byte array elements to a uint and display it.
        public static void ProcessBA(byte[] bytes, int index)
        {


            // Double check for UInt64 Values 


                UInt64 value = BitConverter.ToUInt64(bytes, index);
                Console.WriteLine(formatter, index, BitConverter.ToString(bytes, index, sizeof(UInt64)), value);
            

        }




        static int Main(string[] args)
        {
            Encryption_Module helpers = new Encryption_Module(); 
            bool verbose = false;
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
            UInt64[,] processed_data = process_korea_format(data);
            BinaryWriter file_writer = new BinaryWriter(File.Open(args[1], FileMode.Create));

            // Loop over rows
            for (int i = 0; i <= processed_data.GetUpperBound(0); i++)
                //for (int i = 0; i <= 1; i++)
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
                    byte[] byted_row = new byte[num_cols * sizeof(UInt64)];
                    for (int j = 0; j < num_cols; j++)
                    {
                        Buffer.BlockCopy(BitConverter.GetBytes(processed_data[i,j]), 0, byted_row, j*sizeof(UInt64), sizeof(UInt64));

                    }

                    // For each cell in the row  
                    for (int k = 0; k < num_cols; k++)
                    {
                        if (verbose)
                        {
                            //Console.WriteLine("Before Rand:");
                            ProcessBA(byted_row, k * sizeof(UInt64));

                        }

                        // Pad 2 most significant bytes with random values 
                        RNGCryptoServiceProvider rng = new RNGCryptoServiceProvider();
                        rng.GetBytes(byted_row, sizeof(UInt64) - pad_bytes + k * sizeof(UInt64), pad_bytes); 
                  
                        if (verbose)
                        {
                            //Console.WriteLine("After Rand:");
                            ProcessBA(byted_row, k * sizeof(UInt64));

                        }

                    }

                    //Encrypt Bytes: 
                    UInt64[] encrypted_values = new UInt64[num_cols];
                    for (int current_cell = 0; current_cell < num_cols; current_cell++)
                    {
                        UInt64 c_value_int = BitConverter.ToUInt64(byted_row, sizeof(UInt64) * current_cell);
                        encrypted_values[current_cell] = helpers.encryptCell(c_value_int);
                        Buffer.BlockCopy(BitConverter.GetBytes(encrypted_values[current_cell]), 0, byted_row, current_cell * sizeof(UInt64), sizeof(UInt64));
                        if (verbose)
                        {
                            //Console.WriteLine("After Rand:");
                            ProcessBA(byted_row, current_cell * sizeof(UInt64));

                        }

                    }

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