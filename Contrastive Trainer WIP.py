import torch





def train(model, iterator, optimizer, criterion,duration_loss,sig,clip, device):
    model.train()
    epoch_loss = 0
    
    for src, trg in iterator:
       
        src=src.to(device)
        trg=trg.to(device)
        optimizer.zero_grad()
        output,length, _ = model(src, trg[:,:-1])
        
        output_dim = output.shape[-1]
        
        loss1 = criterion(output.contiguous().view(-1, output_dim),trg[:,1:,:-1].contiguous().view(-1).long())
       
        loss2 = duration_loss(sig(length).contiguous().view(-1).float(),trg[:,1:,-1:].contiguous().view(-1).float())
        
        loss = loss1 + torch.exp(loss2)
        
        loss.backward()
        
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
    
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion,duration_loss,sig,device):
    
    model.eval()
    
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in iterator:
            src=src.to(device)
            trg=trg.to(device)
            
            
            output,length, _ = model(src, trg[:,:-1])
        
            output_dim = output.shape[-1]
        
            loss1 = criterion(output.contiguous().view(-1, output_dim),trg[:,1:,:-1].contiguous().view(-1).long())
       
            loss2 = duration_loss(sig(length).contiguous().view(-1).float(),trg[:,1:,-1:].contiguous().view(-1).float())
        
            loss = loss1 + loss2
                                    
            epoch_loss += loss.item()
        
            
    return epoch_loss / len(iterator)

def accuracy(iterator, model, device,sig, sos_idx , max_len =20):
    model.eval()
    pred=[]
    acc=0
    batch_accuracy=[]
    with torch.no_grad():
        for src,trg in iterator:
                src_tensor=src.to(device)
                trg=trg.to(device)
                src_mask = model.make_src_mask(src_tensor)
                enc_src = model.encoder(src_tensor, src_mask)
                trg_indexes = [[sos_idx,0]]
        
                
                for i in range(max_len):
                    trg_tensor = torch.FloatTensor(trg_indexes).unsqueeze(0).to(device)
                    trg_mask = model.make_trg_mask(trg_tensor).to(device)
                    
                    
                    output,length,attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                    
                    pred_tokens = output.argmax(dim=2)[:,-1].item()
                    pred_len = sig(length[:,-1]).item()
                    trg_indexes.append([pred_tokens,pred_len])
                 
                tar=trg.squeeze().cpu().detach().numpy()
                
                examp_tar=[]
                for item in tar[1:-1]: 
                    duration=round(item[1]*100)
                    for i in range(duration):
                        examp_tar.append(int(item[0]))
               
                
                pred=trg_indexes
                
                examp_pred=[]
                for item in pred[1:-1]: 
                    duration=round(item[1]*100)
                    for i in range(duration):
                        if len(examp_pred)==len(examp_tar):
                            break
                        examp_pred.append(item[0])
                
                correct=0
                for x,y in zip(examp_pred,examp_tar):
                    leng=len(examp_tar)
                    if x==y:
                        correct += 1
                    acc=correct/leng
            
                batch_accuracy.append(acc)
            
    return sum(batch_accuracy)/len(batch_accuracy)
