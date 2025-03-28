# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:52:36 2024

This code is designed to explore the relationship between a star with a planet orbiting it.
This is done by taking in the changes of wavelength of the Balmer line of the star as 
observed on Earth.

A plot of the velocity of the star vs time is produced.



hi, hello
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

COUNT = 0 #The number of lines of information that have been added to the graph

def clean_data(my_df):
    '''

    Parameters
    ----------
    my_df : pandas Dataframe with uncleaned data

    Returns
    -------
    my_df : pandas Dataframe with cleaned data

    '''
    my_df.dropna(subset=['% time (years)'], inplace = True) #Eliminates null values
    my_df.drop(my_df[my_df['% time (years)'].astype(str).str.isalpha()]
               .index, inplace=True) #Deletes all rows with letters in the time field
    my_df.drop(my_df[my_df[' wavelength (nm)'].astype(str).str.isalpha()]
               .index, inplace=True) #Deletes all rows with letters in the wavelength field
    my_df.drop(my_df[abs(my_df[' wavelength (nm)']-656.281)>0.1]
               .index, inplace=True) #Deletes all rows with a
                                        #large difference to the actual wavelength
    my_df.drop(my_df[my_df[' wavelength uncertainty (nm)'].astype(str).str.isalpha()]
               .index, inplace=True) #Deletes all rows with wavelength uncertainty in the time field
    my_df.drop(my_df[my_df[' wavelength uncertainty (nm)']==0]
               .index, inplace=True) #Deletes all rows with wavelength uncertainty equal to 0
    my_df[[
        '% time (years)', ' wavelength (nm)', ' wavelength uncertainty (nm)']] = my_df[[
            '% time (years)', ' wavelength (nm)', ' wavelength uncertainty (nm)']].astype(
                float) #Converts all data to float
    return my_df

def read_file(filename):
    '''

    Parameters
    ----------
    filename : Name of the csv file contiaining data

    Returns
    -------
    my_array : a numpy array containing the data from the csv file
    
    '''
    my_csv = pd.read_csv(f'{filename}.csv') #Converts the csv file into a Dataframe
    my_array = clean_data(my_csv).to_numpy() #Puts cleaned data into an array
    return my_array

def combine_arrays(arr1,arr2):
    '''
    

    Parameters
    ----------
    arr1 : Array of cleaned data
    arr2 : Different array of cleaned data
    Returns
    -------
    my_arr : An array of both previous arrays combined into one

    '''
    my_arr = np.concatenate((arr1, arr2), axis=0)
    return my_arr

def predict_fit(x_val,stellar_speed,angular_velocity):
    '''

    Parameters
    ----------
    x_val : A numpy array of the x values
    stellar_speed : The speed of the star
    angular_velocity : The angular velocity of the star

    Returns
    -------
    y_val : An array of the predicted y values using this model

    '''
    y_val = stellar_speed*np.sin(angular_velocity*x_val*31556736*10**(-8)+np.pi)
    return y_val

def find_anomalies(ydata, prediction, std):
    '''
    
    Parameters
    ----------
    ydata : A numpy array of stellar velocity
    prediction : A numpy array of the predicted values of stellar velocity according to our model
    std : The standard deviation of our model (float)

    Returns
    -------
    to_delete : An array containing the index values of the rows of data that need deleting

    '''

    to_delete = []
    temp = len(ydata)
    for i in range(temp):
        if abs(prediction[i] - ydata[i]) > 3*abs(std):
            to_delete.append(i)
    return to_delete

def delete_rows(arr, indexes):
    ''' 

    Parameters
    ----------
    arr : An array
    indexes : List of indexes that need deleting

    Returns
    -------
    arr : The array with the necessary rows deleted

    '''
    num = 0
    for i in indexes:
        arr = np.delete(arr, i-num, axis = 0)
        num+=1
    return arr

def create_model(x_data,y_data):
    '''
    

    Parameters
    ----------
    x_data : An array containing the x values (Time)
    y_data : An array containing the y values (Stellar velocities)

    Returns
    -------
    variables : The predicted values of Vo and w
    covariance_matrix : The covariance matrix of the 2 variables ([0,0] is Vo and [1,1] is w)

    '''
    variables, covariance_matrix = curve_fit(predict_fit, x_data, y_data, full_output=False)
    return variables, covariance_matrix

def get_stellar_velocity(wavelength, wavelength_uncertainty):
    '''
    

    Parameters
    ----------
    wavelength : A numpy array of the wavelengths observed
    wavelength_uncertainty : A numpy array of the uncertainty on the wavelength observed

    Returns
    -------
    stellar_velocity : A numpy array of the velocities of the star in m/s
    stellar_velocity_uncertainty : A numpy array of the uncertainty on the velocity of the
        star in m/s

    '''
    fractional_wavelength = wavelength/656.281 #Gets lambda(obs)/lambda(0)
    stellar_velocity = (3*10**8)*(fractional_wavelength-1) #v(s) = ((lambda(obs)/lambda(0))-1)*c
    stellar_velocity_uncertainty = wavelength_uncertainty*(3*10**8)/(
        656.281) #delta v(s) = (c/lambda(0)) * delta lambda(obs)
    return stellar_velocity, stellar_velocity_uncertainty

def prediction_data(x_val,stellar_speed,angular_velocity):
    '''
    

    Parameters
    ----------
    x_val : An array containing the require x values
    stellar_speed : The predicted speed of the star
    angular_velocity : The predicted angular velocity of the star

    Returns
    -------
    predicted : An array of the predicted velocity

    '''
    angular_velocity_seconds = 31556736*angular_velocity
    predicted =stellar_speed*np.sin(x_val*angular_velocity_seconds*10**(-8)+np.pi)
    return predicted

def standard_dev(cov_matrix):
    '''
    

    Parameters
    ----------
    cov_matrix : The covariance matrix of Vo and w

    Returns
    -------
    INT
        The standard deviation

    '''
    return cov_matrix[0,1]/(np.sqrt(cov_matrix[0,0])*cov_matrix[1,1])

def plot_graph(x_vals,y_vals, y_err, stellar_speed, angular_velocity):
    '''
    

    Parameters
    ----------
    x_vals : The time in years
    y_vals : The velocity of the star in m/s
    y_err : The uncertainty on the velocity of the star
    stellar_speed : The speed of the star
    angular_velocity : The angular velocity of the star

    Returns
    -------
    None.

    '''
    plt.errorbar(x_vals, y_vals, yerr=y_err, fmt='x', label = 'data')
    plt.ylabel('Stellar velocity (m/s)')
    plt.xlabel('Time (years)')
    plt.title('Stellar velocity vs Time')
    new_x_vals = np.linspace(min(x_vals), max(x_vals),1000)
    new_y_vals = prediction_data(new_x_vals, stellar_speed, angular_velocity)
    plt.plot(new_x_vals,new_y_vals, label = 'Prected data')
    plt.legend(frameon=False, loc='lower right', ncol=2)

def get_chi_squared(y_vals, y_predicted, y_err):
    '''
    

    Parameters
    ----------
    y_vals : The velocity of the star in m/s
    y_predicted : The velocity of the star predicted by the model in m/s
    y_err : The uncertainty on the measured velocity of the star in m/s

    Returns
    -------
    reduced_chi_squared : The reduced chi-squared value of the data
    '''
    chi_squared = 0
    for i in range(len(y_vals)):
        square_difference = (y_vals[i]-y_predicted[i])**2
        chi_squared += square_difference/(y_err[i]**2)
    degrees_of_freedom = len(y_vals)-1
    red_chi_squared = chi_squared/degrees_of_freedom
    return red_chi_squared

def get_distance(angular_velocity, angular_velocity_uncertainty):
    '''
    

    Parameters
    ----------
    angular_velocity : The angular velocity of the Star and Planet (omega) in 10^-8 rad/s
    angular_velocity_uncertainty : The uncertainty angular velocity of the Star and Planet 
        (omega) in 10^-8 rad/s
    Returns
    -------
    final_distance : The distance between the Star and planet in AU
    final_distance_uncertainty : The uncertainty on distance between the Star and planet in AU

    '''
    gravitational_constant = 6.674*10**(-11)
    mass_star = 2.78*1.989*10**30
    distance_cubed = (gravitational_constant*mass_star)/(angular_velocity**2)
    derivative = ((2/3)*gravitational_constant*mass_star*(angular_velocity**(-5)))**(1/3)
    final_distance = distance_cubed**(1/3)/(1.496*10**11)
    final_distance_uncertainty = angular_velocity_uncertainty*derivative/(1.496*10**11)
    return final_distance, final_distance_uncertainty

def get_planet_mass(distance, distance_uncertainty, stellar_speed, steller_speed_uncertainty):
    '''
    

    Parameters
    ----------
    distance : The distance between the Star and planet in AU
    distance_uncertainty : The uncertainty on distance between the Star and planet in AU
    stellar_speed : The speed of the star in m/s
    steller_speed_uncertainty : The uncertainty on the speed of the star in m/s

    Returns
    -------
    INT
        The mass of the planet in Jovian masses

    '''
    gravitational_constant = 6.674*10**(-11)
    mass_star = 2.78*1.989*10**30
    mass_planet = stellar_speed*np.sqrt(mass_star*distance*(1.496*10**11)/gravitational_constant)
    speed_partial_derivative = mass_star*distance*(
        1.496*10**11)/gravitational_constant
    distance_partial_derivative = ((
        stellar_speed**2)*mass_star)/(4*gravitational_constant*distance*(1.496*10**11))
    mass_planet_variance = speed_partial_derivative*(steller_speed_uncertainty**2
                                                     ) + distance_partial_derivative*(
                                                         distance_uncertainty**2)
    mass_planet_uncertainty = np.sqrt(mass_planet_variance)
    return mass_planet/(1.898*10**27), mass_planet_uncertainty/(1.898*10**27)

def round_to_same_dp(num,num_err, sig_fig):
    '''
    

    Parameters
    ----------
    num : The variable
    num_err : The variable's uncertainty
    sig_fig : The number of significant figures that the variable and its uncertainty are rounded to

    Returns
    -------
    rounded_num : The rounded variable
    rounded_num_err : The rounded uncertainty

    '''
    rounded_num = float(f'{num:.{sig_fig}g}')
    rounded_num_err = float(f'{num_err:.{sig_fig}g}')
    return rounded_num, rounded_num_err

def display_variable_on_graph(variable_name, variable, variable_uncertainty = 0, units = ''):
    '''
    

    Parameters
    ----------
    variable_name : The name or symbol of the variable being displayed
    variable : The value of the variable
    variable_uncertainty : The uncertainty on the variable. The default is 0.
    units : The units of the variable. The default is none.

    Displays
    -------
    A line of information onto the graph

    '''
    global COUNT
    COUNT +=1
    position_string = '.' + str(COUNT)
    position = 0.65+float(position_string)/3
    if variable_uncertainty != 0:
        plt.figtext(0.15, position, f"{variable_name} = {variable} ± {variable_uncertainty} {units}"
                    ,fontsize = 8)
    else:
        plt.figtext(0.15, position, f"{variable_name} = {variable} {units}", fontsize = 8)

csv1_data = read_file('doppler_data_1')
csv2_data = read_file('doppler_data_2')
data = combine_arrays(csv1_data, csv2_data)
velocity_star, velocity_star_uncertainty = get_stellar_velocity(data[:,1], data[:,2])
param, param_cov = create_model(data[:,0], velocity_star)
predicted_data = prediction_data(data[:,0], param[0], param[1])
standard_deviation = standard_dev(param_cov)
anomalies = find_anomalies(velocity_star, predicted_data, standard_deviation)
data = delete_rows(data, anomalies)
velocity_star = delete_rows(velocity_star, anomalies)
velocity_star_uncertainty = delete_rows(velocity_star_uncertainty, anomalies)
param, param_cov = create_model(data[:,0], velocity_star)
predicted_data = prediction_data(data[:,0], param[0], param[1])
reduced_chi_squared = get_chi_squared(velocity_star, predicted_data, velocity_star_uncertainty)
orbit_radius, orbit_radius_uncertainty = get_distance(param[1]*10**(-8), param_cov[1,1]*10**(-8))
planet_mass, planet_mass_uncertainty = get_planet_mass(orbit_radius, orbit_radius_uncertainty,
                                                       param[0], param_cov[0,0])
plot_graph(data[:,0], velocity_star, velocity_star_uncertainty, param[0], param[1])
param[0], param_cov[0,0] = round_to_same_dp(param[0], param_cov[0,0], 4)
param[1], param_cov[1,1] = round_to_same_dp(param[1]*10**(-8), param_cov[1,1]*10**(-8), 4)
orbit_radius, orbit_radius_uncertainty = round_to_same_dp(orbit_radius, orbit_radius_uncertainty, 4)
planet_mass, planet_mass_uncertainty = round_to_same_dp(planet_mass, planet_mass_uncertainty, 4)
reduced_chi_squared =round(reduced_chi_squared,3)
display_variable_on_graph(r'$v_{o}$', param[0], param_cov[0,0], 'm/s')
display_variable_on_graph('ω', param[1], param_cov[1,1], 'rad/s')
display_variable_on_graph('r', orbit_radius, orbit_radius_uncertainty, 'AU')
display_variable_on_graph(r'$m_{p}$', planet_mass, planet_mass_uncertainty, 'Jovian masses')
display_variable_on_graph('Reduced $Χ^2$', reduced_chi_squared)
plt.savefig('ChangeInStellarVelocity.png')
plt.show()
