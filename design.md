# 多人2D俯瞰射擊遊戲，但在曲面

## 遊戲概念

- 類似 diep.io 的角色與控制
- 但地圖上每點費賦予了度量張量
- 子彈的軌跡是測地線，而非直線
- 使用socket進行多人連線


### 未來可能的擴充
- 類似魔法的技能，可以改變度量張量

## 系統需求
- 角色控制
- 子彈軌跡
- 房間系統

### 角色控制
使用wasd控制角色移動，滑鼠控制角色朝向。

有三個座標系，世界座標系、局部座標系、朝向座標系。

- 世界座標：實際儲存角色位置的座標系，角色移動時會改變世界座標。
- 局部座標：角色出生時會伴隨一個局部坐標系，由於角色是在曲面上，局部坐標系會隨著角色移動而「平行移動」。
- 朝向座標：角色朝向為上方，角色右側為右方的座標系。

按下wasd時角色會往局部座標上下左右移動。

角色會朝向滑鼠位置，雖然角色朝向應該是角色所處位置的切向量，但是為了操作直覺，我們直接使用滑鼠位置來當作角色朝向。

鏡頭綁定在局部座標係上，也就是鏡頭上方會等於局部座標系的上方，鏡頭右側會等於局部座標系的右側。

### 子彈軌跡
使用滑鼠改變角色朝向，按下滑鼠左鍵發射子彈。子彈軌跡為測地線，會由曲面來計算。

子彈之間相互碰撞會抵銷，子彈與角色碰撞會扣血。

**角色和子彈大小會根據曲面的度量張量來改變。** 

偵測碰撞原本要使用測地線距離，但距離很近的話可以直接用度量張量來計算。

### 房間系統
使用socket進行多人連線，使用者可以創建房間或加入房間，還能選擇房間地圖，甚至自訂地圖的度量張量。

## 物件圖表

<iframe frameborder="0" style="width:100%;height:495px;" src="https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1&title=#R7VtRb9sqFP41eVxlG9txHpus233opGq50n2mMbHRMFiELMl%2B%2FcCGOC7NmqU2ja6ooiocH47hfAe%2Bw4k9AYtq%2F5XDuvzGckQmUZDvJ%2BDzJIriJJP%2FleDQCtIkaAUFx3krCjvBEv9CWmjUtjhHm56iYIwIXPeFK0YpWomeDHLOdn21NSP9u9awQJZguYLElv6Hc1G20iyadvJ%2FEC5Kc%2BcwnbVXnuHqR8HZlur7UUZRe6WCxoye46aEOdudiMDDBCw4Y6L9Vu0XiCivGo%2B1%2Fb6cuXocMkdUXNRBj%2BMnJFs97QXBqnM7PHEw3tjscEWgnAiYrxkVS30lkO1ViUn%2BCA9sq%2B65EXL6pjUvGce%2FpD4k8lIoBfIyFxrsKFXWMCELRhiXgsZTp52Wypi%2BDUcb2e3JzC18IfoG9z3FR7gRZoCMEFhv8HMzZNWxgrzAdM6EYJVW0o5AXKD9WW%2BGR4xk1CNWIcEPUkV3SLU3dcAnWdK2d134hLHWKU9DJ4h12OqQLY6mO%2BzkFw3fGShDC8pJNFeBj5GK%2FHu1fg41spCV8xUNMJz9QC%2BQeAUcSHBBZZOgteqmHIblornXYsFqZayGK0yLx0bnc9xJvutZKxGTfdekCf8S5zmiCjsmoIDPx9iqGaai8Uoylx%2FpvEVwl0wSOfCFbIddW36UOhcLRuVcIG4wQzIKdkhFwoUAB68jrCGVEXsRokbvXYBGHtDRAU0ih4ACD%2BjogE4zh4DGFqBLxOX4PXtewZ4AvKDPWeCSPhO%2FOAdYnPHt0GfqAR0dUKf0OfWAjg6oU%2FrMLEC%2FwsrGz5Pn35PnNHVKnjO%2FNAdYmtnNkGdk14U8oEMD6pI8I18dGh9Ql%2BQZAwsqlBfIkKOhM1EZ9mvVlc6fKUaaZFu%2B0lqRjlFJnQV6wwscESjwz779dwWtXQF7IvDgz9dXpQjRNO6lCDOnKULki18DbEDtiriNFMEufnlAhwbUaYrgK2DjA%2Bo0RYjcpAjpR6YIdpVvueVruPJlhGtyhDDolxEypzX4yFf4htiB0tvJEewKnwd0aECd5gi%2B0Dc%2BoC5zBGAX%2BuZbQpB%2FAOwa%2BoyzPn3Opi7pE%2Fga3wCLE9zOE2DArn95QIcG1CV9Al8EGx9Qp%2FRpHz%2B%2FN0dcRi0MEc3v1WPpssXqxpEnR295UXNiGBmSNNo5hhWj%2Bb8lpi%2F4M4yN4AtWY2zgOD3jS1eXrGAUkodOeuU53zwr3jvnn3ma4OJzvu76pEKo49Eo7vPo1PweaUy049K9OrgsQ3HwhqF2MpahBvfjfC4LBfvwGtzdvRIG0u2P8BkRk6icrKSTgLhoCbfG9QsQ2s6k%2B%2FH2jWV0JgvVPT7pLoOXbIB9KAzf6yXeLusPcNNoXrJPWv%2FfnSVzt7OE%2FQ0hjafX7SwAzO5mp399syarGH6fie0T243vM7OP2Wdi%2Bzh00%2FvMn900kJdks3u7rI2%2B7uU98PAb"></iframe>


client
input->client->server->client->render