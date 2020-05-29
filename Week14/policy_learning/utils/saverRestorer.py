import torch
import os
import glob

class SaverRestorer():
    def __init__(self, config, modelParam):
        self.config      = config
        self.modelParam  = modelParam
        self.save_dir    = self.modelParam['modelsDir']+config['network']+'/'
        self.lowestLoss  = 9999999

        self.removePreviousModel()
        return


    def save(self, update_counter, currentLoss, model):
        #save as the last model
        self.removeLastModel()
        torch.save({
            'update_counter': update_counter,
            'model_state_dict': model.policyNet.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }, self.save_dir+f'last_epoch{update_counter}.pt')

        # save as the best model
        if currentLoss < self.lowestLoss:
            self.removeBestModel()
            self.lowestLoss = currentLoss
            torch.save({
                'update_counter': update_counter,
                'model_state_dict': model.policyNet.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
            }, self.save_dir+f'best_epoch{update_counter}.pt')
        return


    def restore(self, model):
        restore_dir = ''
        paths = glob.glob(self.save_dir + '*')
        if self.modelParam['restoreModelLast'] == 1 and self.modelParam['restoreModelBest'] != 1:
            for path in paths:
                if 'last_epoch' in path:
                    restore_dir = path
        elif self.modelParam['restoreModelLast'] != 1 and self.modelParam['restoreModelBest'] == 1:
            for path in paths:
                if 'best_epoch' in path:
                    restore_dir = path
        if restore_dir!='':
            checkpoint = torch.load(restore_dir)
            model.policyNet.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.update_counter = checkpoint['update_counter'] + 1
        else:
            if self.modelParam['restoreModelLast'] == 1 or self.modelParam['restoreModelBest'] == 1:
                raise ValueError('Could not find the appropriate restore path')
        return model

    def removeLastModel(self):
        files = glob.glob(self.save_dir+'*')
        for f in files:
            if 'last_epoch' in f:
                os.remove(f)
        return

    def removeBestModel(self):
        files = glob.glob(self.save_dir+'*')
        for f in files:
            if 'best_epoch' in f:
                os.remove(f)
        return

    def removePreviousModel(self):
        if self.modelParam['restoreModelLast'] != 1:
            if self.modelParam['restoreModelBest'] != 1:
                # create directory if not existing
                if not os.path.isdir(self.save_dir):
                    os.makedirs(self.save_dir)
                # Remove old files
                files = glob.glob(self.save_dir+'*')
                for f in files:
                    os.remove(f)

