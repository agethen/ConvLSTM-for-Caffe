import datetime
import shutil
import os.path

def ParseProto( filename ):
  # Read in original file
  proto = [line.rstrip('\n') for line in open( filename )]

  # Find block starts
  blocks = []
  for idx, line in enumerate( proto ):
    if line[0:7] == 'message':
      (b, name, _ ) = line.split(' ')
      blocks.append( (idx, 0, name) )

  # EOF
  blocks.append( (len(proto), len(proto), '__eof__' ) )

  # Find actual block-end indicated by }
  for i in range( 0, len(blocks)-1 ):
    cur_line = blocks[i][0]
    next_block_line = blocks[i+1][0]-1

    # Assumption: Correctly formated proto file
    # I.e., indicator of end-of-block is first character in line
    while next_block_line > cur_line:
      if len( proto[next_block_line] ) > 0:
        if proto[next_block_line][0] == "}":
          break
      next_block_line -= 1

    if next_block_line == cur_line:
      print "Warning, could not read block at line ", cur_line, "correctly"

    blocks[i] = ( blocks[i][0], next_block_line, blocks[i][2] )

  return (proto, blocks)




now = datetime.datetime.now()

# Create backup
dst = "src/caffe/proto/caffe.proto." + str( now.year ) + "." + str( now.month ) + "." + str( now.day )
if os.path.isfile( dst ) == False:
  shutil.copyfile( "src/caffe/proto/caffe.proto", dst )
else:
  print "Not creating backup as file already exists"
  print "(File: ", dst, ")"

# Read caffe proto file
(proto, blocks) = ParseProto( "src/caffe/proto/caffe.proto")

# Search block to edit. If it does not exist, create a new one
(patch, patch_blocks) = ParseProto( "patch.proto" )


mod_block_check = []

for i in range( 0, len(patch_blocks)-1 ):
  mod_block_check.append( False )

mod_block_str = []

for i in range( 0, len(patch_blocks)-1 ):
  mod_block_str.append( patch[patch_blocks[i][0] : patch_blocks[i][1]+1] )

output_buffer = []
last = 0

for i in range( 0, len(blocks)-1 ):
  match = False
  for j in range( 0, len(patch_blocks) ):
    # Block of this name already exists. 
    # Add lines
    # TODO(agethen): automatically find correct ids. For now: Static
    if blocks[i][2] == patch_blocks[j][2]:
      mod_block_check[j] = True
      match = True
      output_buffer += proto[last : blocks[i][1]]
      output_buffer += mod_block_str[j][1:-1] # Remove the begin-of-block and end-of-block
      output_buffer += proto[blocks[i][1] : blocks[i+1][0]]

  if match == False:
    output_buffer += proto[last : blocks[i+1][0]]
  last = blocks[i+1][0]

# Write all blocks that did not exist to end of file
for i in range( 0, len(mod_block_check) ):
  if mod_block_check[i] != True:
    output_buffer += mod_block_str[i]

out_file = open( 'src/caffe/proto/caffe.proto', 'w' )

for line in output_buffer:
  out_file.write( line + '\n' )
out_file.close()


print "Successfully edit src/caffe/proto/caffe.proto!"