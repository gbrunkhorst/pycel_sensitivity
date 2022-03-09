import pycel
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns

class InputVar():
    '''Input Variable container and convenience functions'''
    def __init__(self, workbook:pycel.excelcompiler.ExcelCompiler, 
                 sheet:str, cell:str, name= "name", values = [],
                 dist = None, unit = 'Unit'): 
        self.workbook = workbook   
        self.sheet = sheet
        self.cell = cell
        self.name = name
        if self.name=="name":
            self.name = self.cell
        self.values = values
        self.dist = dist  # :scipy.stats. _distn_infrastructure. rv_frozen):
        self.unit = unit
        self.initial_value = self.get_value()
        self.draws = [] 
    def get_value(self):
        '''returns the current value in the workbook'''
        return self.workbook.evaluate(self.sheet+'!'+self.cell)
    def set_value(self, input_value):
        '''sets the input value in the workbook'''
        return self.workbook.set_value(self.sheet+'!'+self.cell, input_value)
    def plot_dist(self, ax):
        #[TODO: work around if no PDF is present]
        x = np.linspace(self.dist.ppf(0.01),
                        self.dist.ppf(0.99), 100)
        ax.plot(x, self.dist.pdf(x))
        ax.set_title(self.name)
        ax.set_xlabel(self.unit)
        ax.set_ylabel('Probability Density')
        return ax
        
class OutputVar():
    '''Output Variable container and convenience functions'''
    def __init__(self, workbook:pycel.excelcompiler.ExcelCompiler, 
                 sheet:str, cell:str, name= "name", 
                  unit = 'Unit'): 
        self.workbook = workbook   
        self.sheet = sheet
        self.cell = cell
        self.name = name
        if self.name=="name":
            self.name = self.cell
        self.draws = []
        self.unit = unit
        self.initial_value = self.get_value()
    def get_value(self):
        '''returns the current value in the workbook'''
        return self.workbook.evaluate(self.sheet+'!'+self.cell)

def plot_hist_ax(var, ax):
    '''Plot a single ax given a variable.  Used in multiple classes.'''
    sns.histplot(var.draws, ax=ax, bins=10, stat='probability')
    ax.set_title(var.name)
    ax.set_xlabel(var.unit)
    ax.set_ylabel('Probability')
    return ax

class Model():
    '''Model container and convenience functions'''
    def __init__(self, workbook:pycel.excelcompiler.ExcelCompiler, 
                 input_vars, output_vars): 
        self.workbook = workbook   
        self.input_vars = input_vars
        self.output_vars = output_vars
    
    def run(self,  iterations = None):
        '''runs the model from a list of inputs or performs monte carlo simulation
        iterations::int if None run inputs as a list  if int, run
            the number of Monte-carlo simulations        
        '''
        # reset the draws
        for iv in self.input_vars:
            iv.draws = []
        for ov in self.output_vars:
            ov.draws = []

        # establish number of rows and draw random values as needed
        if iterations:
            rows = iterations    
            for iv in self.input_vars:
                if iv.dist:
                    iv.draws = iv.dist.rvs(iterations)
                else:
                    iv.draws = np.random.choice(iv.values, size=iterations)
        else:
            rows = len(self.input_vars[0].values) 
            for iv in self.input_vars:
                iv.draws = iv.values
            
        # run the workbook and store results
        for row in range(rows):
            for iv in self.input_vars:
                iv.set_value(iv.draws[row])
            for ov in self.output_vars:
                ov.draws.append(ov.get_value())
    
    def get_io_values(self, input_output_all='all'):
        '''Gets input and output values from the model
        input_output_all::str "input", "output", or "all" grabs the variables indicated
            and returns a dataframe
        '''
        if input_output_all=='input':
            df = self._get_io_values(self.input_vars)
        elif input_output_all=='output':
            df = self._get_io_values(self.output_vars)
        elif input_output_all=='all':
            ivv = self._get_io_values(self.input_vars)
            ivv.columns = pd.MultiIndex.from_tuples([('Inputs', c) for c in ivv.columns])
            ovv = self._get_io_values(self.output_vars)
            ovv.columns = pd.MultiIndex.from_tuples([('Outputs', c) for c in ovv.columns])
            df = pd.concat([ivv, ovv], axis=1)
        else:
            print("input_output_all must equal 'input','output', or 'all'")
        return df

    def _get_io_values(self, vars_list):
        '''Internal helper function'''
        rows = len(vars_list[0].draws)
        columns = len(vars_list)
        col_names = [var.name for var in vars_list]
        df = pd.DataFrame(np.zeros((rows,columns)), 
                    columns = col_names)
        for var, col in zip(vars_list, col_names):
            df[col] = var.draws
        return df

    def plot_hists(self, input_output='input'):    
        '''Histogram plotting function 
        input_output::str "input" or "output" plots the variables indicated
        '''
        if input_output=='input':
            vars_list = self.input_vars
            title = 'Input Histograms'
        elif input_output=='output':
            vars_list = self.output_vars
            title = 'Output Histograms'
            
        fig, axes = plt.subplots(1,len(vars_list), 
                                figsize = (3*len(vars_list),3) ,
                                constrained_layout=True)
        if len(self.input_vars)==1:
            axes = np.array([axes])
        for var, ax in zip(vars_list, axes.flatten()):
            plot_hist_ax(var, ax)
        plt.suptitle(title, size = 16)
        sns.despine()
        return axes
    
    def plot_input_dists(self):
        '''Distribution plotting function'''
        fig, axes = plt.subplots(1,len(self.input_vars), 
                                 figsize = (3*len(self.input_vars),3), 
                                constrained_layout=True)
        if len(self.input_vars)==1:
            axes = np.array([axes])
        for iv, ax in zip(self.input_vars, axes.flatten()):
            iv.plot_dist(ax)
            ax.set_title(iv.name)
        plt.suptitle('Input Distributions', size = 16)
        sns.despine()
        return axes
    
    def describe(self, input_output=None, percentiles = [.1,.5,.9]):
        '''Summarizes the results of the monte carlo simulation'''
        if input_output=='input':
            summary  = self._describe(input_output, percentiles)
        elif input_output=='output':
            summary  = self._describe(input_output, percentiles)
        else:
            summary  = pd.concat([
                self._describe('input', percentiles), 
                self._describe('output', percentiles)],
                axis= 0)
        return summary

    def _describe(self, input_output, percentiles):
        '''Helper function'''
        if input_output=='input':
            vars_list = self.input_vars
        elif input_output=='output':
            vars_list = self.output_vars
            
        summary  = self.get_io_values(input_output).describe(percentiles = [.1,.5,.9]).T
        for var in vars_list:
            summary.at[var.name,'sheet'] = var.sheet
            summary.at[var.name,'cell'] = var.cell
            summary.at[var.name, 'type'] = input_output
            summary.at[var.name,'unit'] = var.unit
        summary = summary[list(summary.columns[-4:])+list(summary.columns[:-4])]
        return summary